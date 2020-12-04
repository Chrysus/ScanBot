import cv2
import datetime
import imutils
import numpy
import time

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
        self.min_area = self.settings.min_roi


        # motion
        self.last_motion_time = None
        self.motion_cooldown = self.settings.motion_cooldown


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
        self.scan_frame_full = None

        # processed frames
        self.cur_frame = None
        self.cur_frame_gray = None
        self.prev_frame = None
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
        self.prev_frame = None
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

    
    def _detect_document(self):
        self._process_cur_frame()
        self._detect_motion()

        if self.motion_detected:
            # image not settled - no document!
            self.document_detected = False
            return

        # TODO - make this work! :)
        self.document_detected = self._calculate_bg_delta()

        return self.document_detected
            

    def _scan_document(self):
        self._process_scan()


    def _detect_motion(self):
        self._calculate_prev_delta()

        # motion detection
        if not self.last_motion_time:
            return
        
        time_delta = time.time() - self.last_motion_time
        if time_delta < self.motion_cooldown:
            # still moving...
            self.motion_detected = True
        else:
            self.motion_detected = False

        return self.motion_detected


        
    def _process_cur_frame(self):
        self.cur_frame = imutils.resize(self.cur_frame_full, width=500)
        self.cur_frame_gray = cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY)
        self.cur_frame_gray = cv2.GaussianBlur(self.cur_frame_gray, (21, 21), 0)

        # set background frames
        if not is_valid_frame(self.bg_frame):
            self.bg_frame = self.cur_frame_gray

        if not is_valid_frame(self.bg_frame_full):
            self.bg_frame_full = self.cur_frame_full.copy()



    def _calculate_bg_delta(self):
        self.bg_delta = self.cur_frame_gray.copy()
        
        frameDelta = cv2.absdiff(self.bg_frame, self.cur_frame_gray)

        thresh = cv2.threshold(frameDelta, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

	# loop over the contours
        for c in contours:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < self.min_area:
                continue
            
            # compute the bounding box for the contour and draw it on the frame
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(self.bg_delta, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # TODO - actually detect a document!
            self.document_detected = True

        return self.document_detected

            
    def _calculate_prev_delta(self):
        #------------------------------------------------
        # TODO - This is too complicated for just simple
        #        motion detection.
        #
        #        Simplify this!
        #------------------------------------------------
        
        if not is_valid_frame(self.prev_frame):
            self.prev_frame = self.cur_frame_gray
            return

        self.prev_delta = self.cur_frame.copy()
        
        prevFrameDelta = cv2.absdiff(self.prev_frame, self.cur_frame_gray)
        
        prevThresh = cv2.threshold(prevFrameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        prevThresh = cv2.dilate(prevThresh, None, iterations=2)
        
        prevContours = cv2.findContours(prevThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        prevContours = imutils.grab_contours(prevContours)

	# loop over the contours
        for c in prevContours:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < self.min_area:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(self.prev_delta, (x, y), (x + w, y + h), (255, 255, 255), 2)

            self.last_motion_time = time.time()
	        
        self.prev_frame = self.cur_frame_gray        


    def _scan(self):
        process_image_height = 500.0
        
        orig = self.cur_frame_full

        image = self.cur_frame_full.copy()
        ratio = image.shape[0] / process_image_height
        image = imutils.resize(image, height = int(process_image_height))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # find the largest contours
        contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

        # process the contours
        document_contours = None
        for c in contours:
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


    def _display(self):
        if self.settings.display == False:
            return
        
        cv2.imshow("Current Frame", self.cur_frame)

        if is_valid_frame(self.bg_delta):
            cv2.imshow("Background Delta", self.bg_delta)

        if is_valid_frame(self.prev_delta):
            cv2.imshow("Prev Frame Delta", self.prev_delta)

        if is_valid_frame(self.document_detect_frame):
            cv2.imshow("Document Detected", imutils.resize(self.document_detect_frame, height = 500))

        if is_valid_frame(self.document_transform_frame):
            cv2.imshow("Scan", self.document_transform_frame)



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
        if not self.last_motion_time:
            return
        
        time_delta = time.time() - self.last_motion_time
        if time_delta < self.motion_cooldown:
            return

        self.scan()


    def scan(self):
        print("Scanning...")
        self.last_motion_time = None

        # check for scan item
        self._scan()

        
# Helper Functions

def is_valid_frame(frame):
    return type(frame) != type(None)

        
def main():
    scanbot = ScanBot()

    scanbot.start()


if __name__ == '__main__':
    main()
