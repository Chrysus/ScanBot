# Settings for ScanBot

class Settings():

    def __init__(self):
        # Display Images
        self.display = True

        # capture resolution
        self.capture_height = 1536
        self.capture_width = 2048

        # Minimum Area of Interest
        self.min_roi = 500

        # Time to Determine when motion stops (seconds)
        self.motion_cooldown = 1.5

        # Time to allow camera auto focus to settle (seconds)
        self.auto_focus_time = 7.0

        # Saving Scans
        self.save_document_scan = True
        self.save_full_image_scan = True
