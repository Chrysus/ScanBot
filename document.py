import cv2
import datetime
import imutils
import numpy
import time

from settings import Settings


class DocumentDetector():
    #-------------------------------
    # init
    #-------------------------------

    def __init__(self):

        # settings
        self.settings = Settings()

        #-------------------------------
        # Settings
        #-------------------------------

        # display
        self.display = self.settings.display
 
        # document
        self.min_roi_area = self.settings.min_roi


        #-------------------------------
        # Internal Data
        #-------------------------------

        self.document_detected = False
        self.motion_detected = False


        #-------------------------------
        # Cached Frames
        #-------------------------------

        # full frames
        self.cur_frame_full = None
        self.bg_frame_full = None
        self.prev_frame_full = None

        # processed frames
        self.cur_frame = None
        self.cur_frame_gray = None

        self.prev_frame = None
        self.prev_frame_gray = None

        self.bg_frame = None
        self.bg_frame_gray = None

        # display frames
        self.bg_delta = None



    def detect_documents(self, frame):
        self._process_frame(frame)
        self._detect_document()

        return self.document_detected


    def _process_frame(self, frame):
        #------------------------------------------------
        # Process the Frame
        #
        # Create the various forms of the frame needed
        # for document detection.
        #------------------------------------------------

        self.cur_frame_full = frame.copy()

        # create a smaller version of the image for faster processing
        self.cur_frame = imutils.resize(self.cur_frame_full, width=500)

        # create a softened (blurred) grayscale version of the smaller image
        self.cur_frame_gray = cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY)
        self.cur_frame_gray = cv2.GaussianBlur(self.cur_frame_gray, (21, 21), 0)

        # set background frames
        # (for now, just do this *once* at startup)
        if not is_valid_frame(self.bg_frame_full):
            self.bg_frame_full = self.cur_frame_full.copy()

        if not is_valid_frame(self.bg_frame):
            self.bg_frame = self.cur_frame_gray
        
        self.delta_display_frame = self.cur_frame.copy()


    def _calculate_bg_delta(self):
        self.bg_delta = self.cur_frame_gray.copy()
        
        frameDelta = cv2.absdiff(self.bg_frame, self.cur_frame_gray)

        thresh = cv2.threshold(frameDelta, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

	# loop over the contours
        for c in contours:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < self.min_roi_area:
                continue
            
            # compute the bounding box for the contour and draw it on the frame
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(self.bg_delta, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # TODO - actually detect a document!
            self.document_detected = True

        return self.document_detected


    def _detect_document(self):
        # HACK
        return self._calculate_bg_delta()



# Helper Functions

def is_valid_frame(frame):
    return type(frame) != type(None)
