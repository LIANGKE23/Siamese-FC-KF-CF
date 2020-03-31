from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k

class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(name='IdentityTracker')

    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box


if __name__ == '__main__':
    # setup tracker
    tracker = IdentityTracker()

    # run experiments on GOT-10k (validation subset)
    experiment = ExperimentGOT10k('E:\PSUThirdSemester\CSE586ComputerVision\Term-Project1\Pythonversion\data\GOT-10k', subset='val')
    experiment.run(tracker, visualize=True)

    # report performance
    experiment.report([tracker.name])