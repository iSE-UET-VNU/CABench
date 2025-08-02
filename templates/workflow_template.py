from scripts.model_inference import model_inference

class Workflow:
    async def __call__(self, data: dict) -> str:
        """
        Process a single testcase and return prediction
        
        Args:
            data: {
                "image_paths": [list of image file paths],
                "video_paths": [list of video file paths], 
                "audio_paths": [list of audio file paths],
                "text_data": {dict of text data}
            }
            
        Returns:
            prediction: string prediction for this testcase
        """
        return ""