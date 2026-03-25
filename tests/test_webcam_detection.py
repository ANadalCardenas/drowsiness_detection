import sys
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


MODEL_PATH = '/workspace/drowsiness_detection/model/exp/weights/best.pt'


def _make_cap(frames, key_sequence, window_props=None):
    """
    Build a mock VideoCapture.

    frames       : list of (ret, frame) pairs returned by cap.read() in order.
    key_sequence : list of values returned by cv2.waitKey() in order.
    window_props : list of values returned by cv2.getWindowProperty() in order
                   (None means always return 1, i.e. window open).
    """
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.side_effect = frames
    return cap


def _fake_frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _wire_model(mock_load):
    fake = _fake_frame()
    mock_results = MagicMock()
    mock_results.render.return_value = [fake]
    mock_model = MagicMock(return_value=mock_results)
    mock_load.return_value = mock_model
    return mock_model, mock_results


@patch('cv2.destroyAllWindows')
@patch('cv2.getWindowProperty')
@patch('cv2.imshow')
@patch('cv2.waitKey')
@patch('cv2.cvtColor')
@patch('cv2.VideoCapture')
@patch('torch.hub.load')
class TestWebcamDetectionMain:

    def test_loads_model_with_correct_path(
        self, mock_load, mock_cap_cls, mock_cvtColor, mock_waitKey,
        mock_imshow, mock_get_prop, mock_destroy
    ):
        """Verifies that torch.hub.load is called with the YOLOv5 custom model
        and the exact path to the pre-trained weights, so the correct model
        is used for real-time inference."""
        mock_model, _ = _wire_model(mock_load)
        frame = _fake_frame()
        mock_cvtColor.return_value = frame

        cap = MagicMock()
        cap.isOpened.side_effect = [True, False]
        cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = cap

        mock_waitKey.return_value = 0
        mock_get_prop.return_value = 1

        from webcam_detection import main
        main()

        mock_load.assert_called_once_with(
            'ultralytics/yolov5',
            'custom',
            path=MODEL_PATH,
            force_reload=True,
        )

    def test_opens_webcam_device_zero(
        self, mock_load, mock_cap_cls, mock_cvtColor, mock_waitKey,
        mock_imshow, mock_get_prop, mock_destroy
    ):
        """Verifies that cv2.VideoCapture is initialised with device index 0,
        which corresponds to the default (first) webcam on the system."""
        _wire_model(mock_load)
        frame = _fake_frame()
        mock_cvtColor.return_value = frame

        cap = MagicMock()
        cap.isOpened.side_effect = [True, False]
        cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = cap

        mock_waitKey.return_value = 0
        mock_get_prop.return_value = 1

        from webcam_detection import main
        main()

        mock_cap_cls.assert_called_once_with(0)

    def test_converts_frame_bgr_to_rgb(
        self, mock_load, mock_cap_cls, mock_cvtColor, mock_waitKey,
        mock_imshow, mock_get_prop, mock_destroy
    ):
        """Verifies that each captured frame is converted from BGR (OpenCV default)
        to RGB before being passed to the YOLO model, which expects RGB input."""
        import cv2
        _wire_model(mock_load)
        frame = _fake_frame()
        mock_cvtColor.return_value = frame

        cap = MagicMock()
        cap.isOpened.side_effect = [True, False]
        cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = cap

        mock_waitKey.return_value = 0
        mock_get_prop.return_value = 1

        from webcam_detection import main
        main()

        mock_cvtColor.assert_called_with(frame, cv2.COLOR_BGR2RGB)

    def test_displays_frame_in_yolo_window(
        self, mock_load, mock_cap_cls, mock_cvtColor, mock_waitKey,
        mock_imshow, mock_get_prop, mock_destroy
    ):
        """Verifies that cv2.imshow is called with the window title 'YOLO',
        displaying the annotated frame with detection overlays in real time."""
        _wire_model(mock_load)
        frame = _fake_frame()
        mock_cvtColor.return_value = frame

        cap = MagicMock()
        cap.isOpened.side_effect = [True, False]
        cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = cap

        mock_waitKey.return_value = 0
        mock_get_prop.return_value = 1

        from webcam_detection import main
        main()

        mock_imshow.assert_called()
        assert mock_imshow.call_args[0][0] == "YOLO"

    def test_exits_loop_on_q_key(
        self, mock_load, mock_cap_cls, mock_cvtColor, mock_waitKey,
        mock_imshow, mock_get_prop, mock_destroy
    ):
        """Verifies that pressing 'q' breaks the capture loop after a single
        frame, providing the user with a keyboard shortcut to stop the detection."""
        _wire_model(mock_load)
        frame = _fake_frame()
        mock_cvtColor.return_value = frame

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = cap

        # 'q' key on the first iteration
        mock_waitKey.return_value = ord('q')
        mock_get_prop.return_value = 1

        from webcam_detection import main
        main()

        # Should have read only one frame before quitting
        cap.read.assert_called_once()

    def test_exits_loop_when_frame_read_fails(
        self, mock_load, mock_cap_cls, mock_cvtColor, mock_waitKey,
        mock_imshow, mock_get_prop, mock_destroy
    ):
        """Verifies that when cap.read() returns ret=False (e.g. the camera is
        disconnected), the loop exits immediately without trying to process the frame."""
        _wire_model(mock_load)

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        mock_cap_cls.return_value = cap

        mock_waitKey.return_value = 0
        mock_get_prop.return_value = 1

        from webcam_detection import main
        main()

        cap.read.assert_called_once()

    def test_exits_loop_when_window_closed(
        self, mock_load, mock_cap_cls, mock_cvtColor, mock_waitKey,
        mock_imshow, mock_get_prop, mock_destroy
    ):
        """Verifies that closing the display window manually (getWindowProperty
        returns a negative value) also stops the capture loop, preventing the
        script from running headlessly in the background."""
        _wire_model(mock_load)
        frame = _fake_frame()
        mock_cvtColor.return_value = frame

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = cap

        mock_waitKey.return_value = 0
        # Negative value signals that the window has been closed
        mock_get_prop.return_value = -1

        from webcam_detection import main
        main()

        cap.read.assert_called_once()

    def test_exits_loop_when_get_window_property_raises(
        self, mock_load, mock_cap_cls, mock_cvtColor, mock_waitKey,
        mock_imshow, mock_get_prop, mock_destroy
    ):
        """Verifies that a cv2.error raised by getWindowProperty (which can happen
        when the window no longer exists) is caught and used as a signal to exit
        the loop gracefully instead of crashing."""
        import cv2
        _wire_model(mock_load)
        frame = _fake_frame()
        mock_cvtColor.return_value = frame

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = cap

        mock_waitKey.return_value = 0
        mock_get_prop.side_effect = cv2.error

        from webcam_detection import main
        main()

        cap.read.assert_called_once()

    def test_releases_capture_on_exit(
        self, mock_load, mock_cap_cls, mock_cvtColor, mock_waitKey,
        mock_imshow, mock_get_prop, mock_destroy
    ):
        """Verifies that cap.release() is called when the loop ends, freeing the
        webcam device so other applications can use it after detection stops."""
        _wire_model(mock_load)
        frame = _fake_frame()
        mock_cvtColor.return_value = frame

        cap = MagicMock()
        cap.isOpened.side_effect = [True, False]
        cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = cap

        mock_waitKey.return_value = 0
        mock_get_prop.return_value = 1

        from webcam_detection import main
        main()

        cap.release.assert_called_once()

    def test_destroys_windows_on_exit(
        self, mock_load, mock_cap_cls, mock_cvtColor, mock_waitKey,
        mock_imshow, mock_get_prop, mock_destroy
    ):
        """Verifies that cv2.destroyAllWindows() is called when the loop ends,
        ensuring all OpenCV GUI windows are properly closed on exit."""
        _wire_model(mock_load)
        frame = _fake_frame()
        mock_cvtColor.return_value = frame

        cap = MagicMock()
        cap.isOpened.side_effect = [True, False]
        cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = cap

        mock_waitKey.return_value = 0
        mock_get_prop.return_value = 1

        from webcam_detection import main
        main()

        mock_destroy.assert_called_once()
