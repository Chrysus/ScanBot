# Settings for ScanBot

class Settings():

    def __init__(self):
        # Display Images
        self.display = True

        # capture resolution
        self.capture_height = 1536
        self.capture_width = 2048

        # processing resolution
        self.processing_height = 480
        self.processing_width = 640

        # Minimum Area of Interest
        self.min_roi = 500

        # Minimum Motion Area
        self.min_motion_area = 400

        # Time to Determine when motion stops (seconds)
        self.motion_cooldown = 1.5

        # Time to allow camera auto focus to settle (seconds)
        self.auto_focus_time = 7.0

        # Saving Scans
        self.save_document_scan = True
        self.save_full_image_scan = True
