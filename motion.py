import cv2
import datetime
import imutils
import numpy
import time

from settings import Settings


class MotionDetector():

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
 
        # motion
        self.last_motion_time = None
        self.motion_cooldown = self.settings.motion_cooldown
        self.min_motion_area = self.settings.min_motion_area


        #-------------------------------
        # Internal Data
        #-------------------------------

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
        self.delta_display_frame = None



    def detect_motion(self, frame):
        self._process_Frame(frame)
        self._detect_motion()

        # motion detection
        if not self.last_motion_time:
            return False
        
        time_delta = time.time() - self.last_motion_time
        if time_delta < self.motion_cooldown:
            # still moving...
            self.motion_detected = True
        else:
            self.motion_detected = False

        return self.motion_detected


    def display_frames(self):
        # Previous Frame Delta
        if is_valid_frame(self.delta_display_frame):
            cv2.imshow("Motion Frame", self.delta_display_frame)




    def _process_Frame(self, frame):
        #------------------------------------------------
        # Process the Frame
        #
        # Create the various forms of the frame needed
        # for motion detection.
        #------------------------------------------------

        self.cur_frame_full = frame.copy()

        # cache the prev frame
        self.prev_frame_gray = self.cur_frame_gray
        
        # create a smaller version of the image for faster processing
        self.cur_frame = imutils.resize(self.cur_frame_full, width=500)

        # create a softened (blurred) grayscale version of the smaller image
        self.cur_frame_gray = cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY)
        self.cur_frame_gray = cv2.GaussianBlur(self.cur_frame_gray, (21, 21), 0)
        
        if not is_valid_frame(self.prev_frame_gray):
            self.prev_frame_gray = self.cur_frame_gray
            return

        self.delta_display_frame = self.cur_frame.copy()
        

        
    def _detect_motion(self):

        #------------------------------------------------
        # TODO - This is too complicated for just simple
        #        motion detection.
        #
        #        Simplify this!
        #------------------------------------------------

        self.delta_display_frame = self.cur_frame.copy()
        
        delta_frame = cv2.absdiff(self.prev_frame_gray, self.cur_frame_gray)
        
        thresh_delta_frame = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
        thresh_delta_frame = cv2.dilate(thresh_delta_frame, None, iterations=2)
        
        contours = cv2.findContours(thresh_delta_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        motion_detected = False
        
	# loop over the contours
        for c in contours:
            if cv2.contourArea(c) >= self.min_motion_area:
                motion_detected = True
                
                if self.display:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(self.delta_display_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                else:
                    break

        if motion_detected:
            self.last_motion_time = time.time()
	        


# Helper Functions

def is_valid_frame(frame):
    return type(frame) != type(None)
