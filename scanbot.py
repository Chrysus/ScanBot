import cv2
import datetime
import imutils
import numpy
import time

from document import DocumentDetector
from motion import MotionDetector
from settings import Settings
from transform import four_point_transform


class ScanBot():

    #-------------------------------
    # init
    #-------------------------------

    def __init__(self):
        self.cam = None

        # settings
        self.settings = Settings()

        #-------------------------------
        # Settings
        #-------------------------------

        # display
        self.display = self.settings.display
 
        # scan
        self.min_roi_area = self.settings.min_roi

        # motion
        self.motion_detector = MotionDetector()
        self.min_motion_area = self.settings.min_motion_area

        # document
        self.document_detector = DocumentDetector()
        
        # storage
        self.save_document_scan = self.settings.save_document_scan
        self.save_full_image_scan = self.settings.save_full_image_scan
        self.store_document_callback = self._store_document
        self.store_full_image_callback = self._store_full_image


        #-------------------------------
        # Internal Data
        #-------------------------------

        self.motion_detected = False
        self.document_detected = False
        self.document_scanned = False
        
        #-------------------------------
        # Cached Frames
        #-------------------------------

        # full frames
        self.cur_frame_full = None
        self.bg_frame_full = None
        self.scan_frame_full = None

        # processed frames
        self.cur_frame = None
        self.cur_frame_gray = None
        self.prev_frame_gray = None
        self.bg_frame = None
        self.scan_frame_transform = None

        # display frames
        self.bg_delta = None
        self.prev_delta = None
        self.document_detect_frame = None
        self.document_transform_frame = None



    #-------------------------------
    # Public Methods
    #-------------------------------    

    # Start
    
    def start(self):
        self._start_camera()

        done = False
        
        while not done:
            frame = self._capture_frame()

            if is_valid_frame(frame):
                self.cur_frame_full = frame

                # Detect
                self._detect_document()

                # Scan
                if self.document_detected:
                    self._scan_document()

                # Display
                self._display()

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                done = True

        self.stop()


    # Stop
        
    def stop(self):
        self._stop_camera()
        cv2.destroyAllWindows()

        

    #-------------------------------
    # Private Methods
    #-------------------------------    

    def _start_camera(self):
        # These are just some example resolutons
        resolutions = [(640, 480), (800, 600), (1024, 768), (1280, 720), (1600, 1200)]
        resolutions.extend([(1920, 1080), (2048, 1536), (2592, 1944), (3264, 1836), (3264, 2448)])

        width = self.settings.capture_width
        height = self.settings.capture_height

        # TODO - handle camera init failure
        
        self.cam = cv2.VideoCapture(-1)

        if self.cam == None:
            print("ERROR - cannot find camera!")
            return
        
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        print("Starting camera...")

        self._auto_focus()
                        

    def _stop_camera(self):
        self.cur_frame = None
        self.cur_frame_gray = None
        self.prev_frame_gray = None
        self.bg_frame = None

        self.bg_delta = None
        self.prev_delta = None

        if self.cam != None:
            self.cam.release()
            self.cam = None

            
    def _capture_frame(self):
        frame = None
        if self.cam != None:
            _, frame = self.cam.read()

        return frame

    #-----------------------------------------------------
    # Detect New Document
    #-----------------------------------------------------
    
    def _detect_document(self):
        self._process_cur_frame()
        self._detect_motion()

        #-------------------------------------------------
        # motion is a proxy for: "hey! scan this!"
        #
        # motion indicates the possible removal of the
        # previously scanned document and/or the insertion
        # of a new document to be scanned.
        #-------------------------------------------------

        if self.motion_detected:
            # image not settled - no document!
            self.document_detected = False

            # reset the document_scanned flag
            self.document_scanned = False
            return

        # TODO - make this work! :)
        self.document_detected = self.document_detector.detect_documents(self.cur_frame_full)

        return self.document_detected
            

    #-----------------------------------------------------
    # Scan New Document
    #-----------------------------------------------------

    def _scan_document(self):
        self._process_scan()


    #-----------------------------------------------------
    # Display
    #-----------------------------------------------------

    def _display(self):
        if self.settings.display == False:
            return

        # Current Frame
        cv2.imshow("Current Frame", self.cur_frame)

        # Background Delta
        if is_valid_frame(self.bg_delta):
            cv2.imshow("Background Delta", self.bg_delta)

        # Motion Frame
        self.motion_detector.display_frames()
            
        # Previous Frame Delta
        if is_valid_frame(self.prev_delta):
            cv2.imshow("Prev Frame Delta", self.prev_delta)

        # Detected Document
        if is_valid_frame(self.document_detect_frame):
            cv2.imshow("Document Detected", imutils.resize(self.document_detect_frame, height = 500))

        # Scanned Document
        if is_valid_frame(self.document_transform_frame):
            cv2.imshow("Scan", self.document_transform_frame)

        
    #-----------------------------------------------------
    # Helper Methods
    #-----------------------------------------------------

    def _process_cur_frame(self):
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


    def _detect_motion(self):
        self.motion_detected = self.motion_detector.detect_motion(self.cur_frame_full)
        return self.motion_detected
        

    def _scan(self):
        process_image_height = 500.0
        
        orig = self.cur_frame_full

        image = self.cur_frame_full.copy()
        ratio = image.shape[0] / process_image_height
        image = imutils.resize(image, height = int(process_image_height))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # DEBUG
        DEBUG_DISPLAY = False
        if DEBUG_DISPLAY:
            if is_valid_frame(edged):
                cv2.imshow("DEBUG SCAN - Edged", edged)


        # find the largest contours
        contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

        # process the contours
        document_contours = None
        for c in contours:
            # ignore contours that are too small
            if cv2.contourArea(c) <= self.min_roi_area:
                continue

	    contour_length = cv2.arcLength(c, True)
	    approx_poly = cv2.approxPolyDP(c, 0.02 * contour_length, True)

	    # if approximated poly has four points then...document?
	    if len(approx_poly) == 4:
		document_contours = approx_poly
		break

        if type(document_contours) == type(None):
            print("Document Not Found")
            return

        # draw the contours of the document
        cv2.drawContours(image, [document_contours], -1, (0, 255, 0), 2)
        self.document_detect_frame = image

        # finally, transform the document (i.e. remove rotation)
        document_transform_frame = four_point_transform(orig, document_contours.reshape(4, 2) * ratio)
        self.document_transform_frame = document_transform_frame


    def _auto_focus(self):
        # if the camera has auto fucus
        focus_time = self.settings.auto_focus_time
        start_time = time.time()
        cur_time = time.time()
        
        delta = cur_time - start_time
        while (delta <= focus_time):
            self.cam.read()
            cur_time = time.time()
            delta = cur_time - start_time


    def _process_scan(self):
        if self.document_scanned:
            # this document has already been scanned!
            return
        
        self.scan()


    #------------------------------------------------
    # Storage
    #------------------------------------------------

    def _store_document(image):
        pass

    
    def _store_full_image(image):
        pass



    def scan(self):
        print("Scanning...")

        # check for scan item
        self._scan()
        self.document_scanned = True

        
# Helper Functions

def is_valid_frame(frame):
    return type(frame) != type(None)

        
def main():
    scanbot = ScanBot()

    scanbot.start()


if __name__ == '__main__':
    main()
