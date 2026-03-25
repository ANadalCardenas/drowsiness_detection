import sys
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import training_set_generation as tsg


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestConstants:

    def test_labels_contains_awake(self):
        """Verifies that 'awake' is one of the recognised class labels, since the
        model must be able to distinguish alert drivers from drowsy ones."""
        assert 'awake' in tsg.LABELS

    def test_labels_contains_drowsy(self):
        """Verifies that 'drowsy' is one of the recognised class labels, which is
        the primary detection target of the drowsiness detection system."""
        assert 'drowsy' in tsg.LABELS

    def test_labels_has_exactly_two_entries(self):
        """Verifies that only two classes are defined ('awake' and 'drowsy'),
        matching the binary classification task the model is trained for."""
        assert len(tsg.LABELS) == 2

    def test_num_imgs_is_two(self):
        """Verifies that NUM_IMGS is set to 2, controlling how many images are
        collected per label during a single training-data capture session."""
        assert tsg.NUM_IMGS == 2

    def test_images_path_ends_with_data_test(self):
        """Verifies that collected images are stored under the 'data/test'
        subdirectory of the project root, as expected by the training pipeline."""
        assert tsg.IMAGES_PATH.endswith(os.path.join("data", "test"))

    def test_project_root_is_parent_of_scripts(self):
        """Verifies that PROJECT_ROOT is correctly resolved as the parent directory
        of the 'scripts' folder, ensuring all derived paths point to the right
        location regardless of where the script is invoked from."""
        scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        expected_root = os.path.dirname(os.path.abspath(scripts_dir))
        assert os.path.abspath(tsg.PROJECT_ROOT) == expected_root


# ---------------------------------------------------------------------------
# main() behaviour
# ---------------------------------------------------------------------------

def _fake_frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _build_cap(key_sequence, frame=None):
    """
    Return a mock VideoCapture whose waitKey returns values from key_sequence.
    cap.read() always succeeds with `frame` (or a zero-filled array).
    """
    if frame is None:
        frame = _fake_frame()
    cap = MagicMock()
    cap.read.return_value = (True, frame)
    return cap


@patch('time.sleep')
@patch('cv2.destroyAllWindows')
@patch('cv2.imwrite')
@patch('cv2.imshow')
@patch('cv2.waitKey')
@patch('cv2.VideoCapture')
@patch('os.makedirs')
class TestTrainingSetGenerationMain:

    def _run_with_keys(self, mock_makedirs, mock_cap_cls, mock_waitKey,
                       mock_imshow, mock_imwrite, mock_destroy, mock_sleep,
                       key_sequence):
        """Helper that sets up a cap whose waitKey yields key_sequence."""
        frame = _fake_frame()
        cap = _build_cap(key_sequence, frame)
        mock_cap_cls.return_value = cap
        mock_waitKey.side_effect = key_sequence
        tsg.main()
        return cap, frame

    # --- directory creation ---------------------------------------------------

    def test_creates_images_directory_for_each_label(
        self, mock_makedirs, mock_cap_cls, mock_waitKey,
        mock_imshow, mock_imwrite, mock_destroy, mock_sleep
    ):
        """Verifies that os.makedirs is called for the images output directory
        once per label, ensuring the destination folder exists before any image
        is written to disk."""
        # Spacebar twice per label (NUM_IMGS = 2), for both labels
        keys = [ord(' '), ord(' ')] * 2 + [0] * 50
        self._run_with_keys(
            mock_makedirs, mock_cap_cls, mock_waitKey,
            mock_imshow, mock_imwrite, mock_destroy, mock_sleep,
            keys,
        )
        images_path_calls = [
            c for c in mock_makedirs.call_args_list
            if c[0][0] == tsg.IMAGES_PATH
        ]
        assert len(images_path_calls) == len(tsg.LABELS)

    def test_creates_labels_directory_after_collection(
        self, mock_makedirs, mock_cap_cls, mock_waitKey,
        mock_imshow, mock_imwrite, mock_destroy, mock_sleep
    ):
        """Verifies that a 'labels' directory is created at the same level as the
        images directory after collection completes, ready for the user to place
        Label Studio annotation files."""
        keys = [ord(' '), ord(' ')] * 2 + [0] * 50
        self._run_with_keys(
            mock_makedirs, mock_cap_cls, mock_waitKey,
            mock_imshow, mock_imwrite, mock_destroy, mock_sleep,
            keys,
        )
        labels_path = os.path.join(os.path.dirname(tsg.IMAGES_PATH), "labels")
        labels_calls = [
            c for c in mock_makedirs.call_args_list
            if c[0][0] == labels_path
        ]
        assert len(labels_calls) == 1

    # --- image saving --------------------------------------------------------

    def test_saves_correct_number_of_images_per_label(
        self, mock_makedirs, mock_cap_cls, mock_waitKey,
        mock_imshow, mock_imwrite, mock_destroy, mock_sleep
    ):
        """Verifies that exactly NUM_IMGS images are saved for each label,
        totalling NUM_IMGS * len(LABELS) writes, so the dataset is balanced
        across classes."""
        keys = [ord(' '), ord(' ')] * 2 + [0] * 50
        self._run_with_keys(
            mock_makedirs, mock_cap_cls, mock_waitKey,
            mock_imshow, mock_imwrite, mock_destroy, mock_sleep,
            keys,
        )
        # NUM_IMGS images for each of the 2 labels
        assert mock_imwrite.call_count == tsg.NUM_IMGS * len(tsg.LABELS)

    def test_image_filename_contains_label(
        self, mock_makedirs, mock_cap_cls, mock_waitKey,
        mock_imshow, mock_imwrite, mock_destroy, mock_sleep
    ):
        """Verifies that every saved filename starts with the corresponding class
        label (e.g. 'awake.' or 'drowsy.'), allowing the training pipeline to
        identify the class from the filename alone."""
        keys = [ord(' '), ord(' ')] * 2 + [0] * 50
        self._run_with_keys(
            mock_makedirs, mock_cap_cls, mock_waitKey,
            mock_imshow, mock_imwrite, mock_destroy, mock_sleep,
            keys,
        )
        saved_paths = [c[0][0] for c in mock_imwrite.call_args_list]
        for label in tsg.LABELS:
            matching = [p for p in saved_paths if os.path.basename(p).startswith(label + '.')]
            assert len(matching) == tsg.NUM_IMGS, (
                f"Expected {tsg.NUM_IMGS} images for label '{label}', got {len(matching)}"
            )

    def test_image_saved_in_correct_directory(
        self, mock_makedirs, mock_cap_cls, mock_waitKey,
        mock_imshow, mock_imwrite, mock_destroy, mock_sleep
    ):
        """Verifies that all images are written to IMAGES_PATH (data/test),
        ensuring collected frames end up in the directory expected by the
        training and labelling workflow."""
        keys = [ord(' '), ord(' ')] * 2 + [0] * 50
        self._run_with_keys(
            mock_makedirs, mock_cap_cls, mock_waitKey,
            mock_imshow, mock_imwrite, mock_destroy, mock_sleep,
            keys,
        )
        saved_paths = [c[0][0] for c in mock_imwrite.call_args_list]
        for path in saved_paths:
            assert os.path.dirname(path) == tsg.IMAGES_PATH

    def test_image_filename_ends_with_jpg(
        self, mock_makedirs, mock_cap_cls, mock_waitKey,
        mock_imshow, mock_imwrite, mock_destroy, mock_sleep
    ):
        """Verifies that every saved image file has a .jpg extension, matching
        the format expected by Label Studio and the YOLOv5 training pipeline."""
        keys = [ord(' '), ord(' ')] * 2 + [0] * 50
        self._run_with_keys(
            mock_makedirs, mock_cap_cls, mock_waitKey,
            mock_imshow, mock_imwrite, mock_destroy, mock_sleep,
            keys,
        )
        saved_paths = [c[0][0] for c in mock_imwrite.call_args_list]
        for path in saved_paths:
            assert path.endswith('.jpg')

    # --- early exit on 'q' ---------------------------------------------------

    def test_q_key_stops_collection_early(
        self, mock_makedirs, mock_cap_cls, mock_waitKey,
        mock_imshow, mock_imwrite, mock_destroy, mock_sleep
    ):
        """Verifies that pressing 'q' immediately aborts the collection loop
        without saving any images, giving the user an escape hatch at any point
        during the session."""
        # Press 'q' on the very first frame — no images should be saved
        keys = [ord('q')] + [0] * 50
        self._run_with_keys(
            mock_makedirs, mock_cap_cls, mock_waitKey,
            mock_imshow, mock_imwrite, mock_destroy, mock_sleep,
            keys,
        )
        mock_imwrite.assert_not_called()

    def test_q_key_releases_cap_and_destroys_windows(
        self, mock_makedirs, mock_cap_cls, mock_waitKey,
        mock_imshow, mock_imwrite, mock_destroy, mock_sleep
    ):
        """Verifies that when the user quits with 'q', the webcam is released and
        all OpenCV windows are destroyed, preventing resource leaks on early exit."""
        frame = _fake_frame()
        cap = _build_cap([], frame)
        mock_cap_cls.return_value = cap
        mock_waitKey.return_value = ord('q')

        tsg.main()

        cap.release.assert_called()
        mock_destroy.assert_called()

    # --- webcam lifecycle ----------------------------------------------------

    def test_webcam_opened_once_per_label(
        self, mock_makedirs, mock_cap_cls, mock_waitKey,
        mock_imshow, mock_imwrite, mock_destroy, mock_sleep
    ):
        """Verifies that a fresh VideoCapture is opened for each label, so the
        camera is re-initialised between collection rounds and the user has time
        to reposition before capturing images for the next class."""
        keys = [ord(' '), ord(' ')] * 2 + [0] * 50
        self._run_with_keys(
            mock_makedirs, mock_cap_cls, mock_waitKey,
            mock_imshow, mock_imwrite, mock_destroy, mock_sleep,
            keys,
        )
        assert mock_cap_cls.call_count == len(tsg.LABELS)

    def test_webcam_released_after_each_label(
        self, mock_makedirs, mock_cap_cls, mock_waitKey,
        mock_imshow, mock_imwrite, mock_destroy, mock_sleep
    ):
        """Verifies that cap.release() is called exactly once for each label's
        VideoCapture, ensuring the camera device is freed between collection
        rounds and not kept open unnecessarily."""
        caps = [MagicMock() for _ in tsg.LABELS]
        for cap in caps:
            cap.read.return_value = (True, _fake_frame())
        mock_cap_cls.side_effect = caps

        keys = [ord(' '), ord(' ')] * 2 + [0] * 50
        mock_waitKey.side_effect = keys

        tsg.main()

        for cap in caps:
            cap.release.assert_called_once()

    # --- warm-up sleep -------------------------------------------------------

    def test_sleeps_for_warmup_after_opening_webcam(
        self, mock_makedirs, mock_cap_cls, mock_waitKey,
        mock_imshow, mock_imwrite, mock_destroy, mock_sleep
    ):
        """Verifies that time.sleep(2) is called after opening the webcam, giving
        the camera sensor time to adjust its exposure before frames are captured
        for the dataset."""
        keys = [ord(' '), ord(' ')] * 2 + [0] * 50
        self._run_with_keys(
            mock_makedirs, mock_cap_cls, mock_waitKey,
            mock_imshow, mock_imwrite, mock_destroy, mock_sleep,
            keys,
        )
        # time.sleep(2) is called for warm-up and for pause between labels
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert 2 in sleep_calls

    # --- failed frame read ---------------------------------------------------

    def test_handles_failed_frame_read(
        self, mock_makedirs, mock_cap_cls, mock_waitKey,
        mock_imshow, mock_imwrite, mock_destroy, mock_sleep
    ):
        """Verifies that when cap.read() returns ret=False (e.g. the camera is
        unavailable), the collection loop exits without raising an exception and
        without writing any images to disk."""
        cap = MagicMock()
        cap.read.return_value = (False, None)
        mock_cap_cls.return_value = cap
        mock_waitKey.return_value = 0

        # Should not raise; collection just ends gracefully
        tsg.main()

        mock_imwrite.assert_not_called()
