import sys
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


MODEL_PATH = '/workspace/drowsiness_detection/model/exp/weights/best.pt'
IMAGE_PATH = '/workspace/drowsiness_detection/data/test/awake.3e3befe5-a20e-11f0-b1eb-1d37d0d6c335.jpg'


def _make_mocks():
    """Return a (mock_model, mock_results) pair wired for a successful run."""
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_results = MagicMock()
    mock_results.render.return_value = [fake_frame]

    mock_model = MagicMock(return_value=mock_results)
    return mock_model, mock_results, fake_frame


@patch('cv2.destroyAllWindows')
@patch('cv2.waitKey')
@patch('cv2.imshow')
@patch('cv2.cvtColor')
@patch('cv2.imread')
@patch('torch.hub.load')
class TestImageDetectionMain:

    def test_loads_model_with_correct_path(
        self, mock_load, mock_imread, mock_cvtColor, mock_imshow, mock_waitKey, mock_destroy
    ):
        """Verifies that torch.hub.load is called with the YOLOv5 custom model
        and the exact path to the pre-trained weights file."""
        mock_model, mock_results, fake_frame = _make_mocks()
        mock_load.return_value = mock_model
        mock_imread.return_value = fake_frame
        mock_cvtColor.return_value = fake_frame

        from image_detection import main
        main()

        mock_load.assert_called_once_with(
            'ultralytics/yolov5',
            'custom',
            path=MODEL_PATH,
            force_reload=True,
        )

    def test_reads_image_from_correct_path(
        self, mock_load, mock_imread, mock_cvtColor, mock_imshow, mock_waitKey, mock_destroy
    ):
        """Verifies that cv2.imread is called with the hardcoded test image path,
        ensuring the script targets the expected input file."""
        mock_model, mock_results, fake_frame = _make_mocks()
        mock_load.return_value = mock_model
        mock_imread.return_value = fake_frame
        mock_cvtColor.return_value = fake_frame

        from image_detection import main
        main()

        mock_imread.assert_called_once_with(IMAGE_PATH)

    def test_converts_bgr_to_rgb(
        self, mock_load, mock_imread, mock_cvtColor, mock_imshow, mock_waitKey, mock_destroy
    ):
        """Verifies that the image loaded by OpenCV (BGR format) is converted
        to RGB before being passed to the YOLO model, which expects RGB input."""
        import cv2
        mock_model, mock_results, fake_frame = _make_mocks()
        mock_load.return_value = mock_model
        mock_imread.return_value = fake_frame
        mock_cvtColor.return_value = fake_frame

        from image_detection import main
        main()

        mock_cvtColor.assert_called_once_with(fake_frame, cv2.COLOR_BGR2RGB)

    def test_runs_inference_on_converted_image(
        self, mock_load, mock_imread, mock_cvtColor, mock_imshow, mock_waitKey, mock_destroy
    ):
        """Verifies that the model is called with the RGB-converted image (not
        the raw BGR one), confirming the correct data flows into inference."""
        mock_model, mock_results, fake_frame = _make_mocks()
        rgb_frame = np.ones((100, 100, 3), dtype=np.uint8)
        mock_load.return_value = mock_model
        mock_imread.return_value = fake_frame
        mock_cvtColor.return_value = rgb_frame

        from image_detection import main
        main()

        mock_model.assert_called_once_with(rgb_frame)

    def test_prints_results(
        self, mock_load, mock_imread, mock_cvtColor, mock_imshow, mock_waitKey, mock_destroy
    ):
        """Verifies that results.print() is called after inference, so detection
        output (bounding boxes, labels, confidence scores) is logged to the console."""
        mock_model, mock_results, fake_frame = _make_mocks()
        mock_load.return_value = mock_model
        mock_imread.return_value = fake_frame
        mock_cvtColor.return_value = fake_frame

        from image_detection import main
        main()

        mock_results.print.assert_called_once()

    def test_shows_annotated_image(
        self, mock_load, mock_imread, mock_cvtColor, mock_imshow, mock_waitKey, mock_destroy
    ):
        """Verifies that cv2.imshow is called with the window title 'Annotated',
        displaying the rendered detection overlays to the user."""
        mock_model, mock_results, fake_frame = _make_mocks()
        mock_load.return_value = mock_model
        mock_imread.return_value = fake_frame
        mock_cvtColor.return_value = fake_frame

        from image_detection import main
        main()

        mock_imshow.assert_called_once()
        window_name = mock_imshow.call_args[0][0]
        assert window_name == "Annotated"

    def test_waits_for_key_press(
        self, mock_load, mock_imread, mock_cvtColor, mock_imshow, mock_waitKey, mock_destroy
    ):
        """Verifies that cv2.waitKey(0) is called after displaying the image,
        blocking execution until the user presses any key to close the window."""
        mock_model, mock_results, fake_frame = _make_mocks()
        mock_load.return_value = mock_model
        mock_imread.return_value = fake_frame
        mock_cvtColor.return_value = fake_frame

        from image_detection import main
        main()

        mock_waitKey.assert_called_once_with(0)

    def test_destroys_windows_after_key_press(
        self, mock_load, mock_imread, mock_cvtColor, mock_imshow, mock_waitKey, mock_destroy
    ):
        """Verifies that cv2.destroyAllWindows() is called at the end of main(),
        cleaning up all OpenCV GUI windows after the user dismisses the display."""
        mock_model, mock_results, fake_frame = _make_mocks()
        mock_load.return_value = mock_model
        mock_imread.return_value = fake_frame
        mock_cvtColor.return_value = fake_frame

        from image_detection import main
        main()

        mock_destroy.assert_called_once()
