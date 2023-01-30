from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--savename', type=str, default='Real', help='the name of saved folder')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--gamma_input_test', action='store_true', help='if add gamma to input image for testing')
        self.parser.add_argument('--savelight_to_multi', action='store_true', help='save light into seperate txt file')
        self.parser.add_argument('--savelight_to_one', action='store_true', help='save light into one txt file')
        self.isTrain = False
