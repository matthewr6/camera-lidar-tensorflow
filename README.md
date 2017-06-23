### Gathering and formatting data

- Create rosbag of data (`rosbag record -a`)
- Copy rosbag into this directory (`scp`)
- Run `python parsebag.py <name>` in this folder, where `<name>` is the name of your rosbag file (`something.bag`)
- `cp dict_data/* normalized_data/*`
- `python ackermann_normalizing.py`


### Training

- `python laser_cnn_cdf.py`
	- overnight


### Using on robot:

- Copy files in `models/` to `~/usuavcteam/iee-avc/src/avc/nodes/steering_utils/model` on the Pi
- Run the autosteering ndoe along with roscore and the sensors + teleop roslaunch files


### Checking network accuracy

- `python confusion_matrix test_cdf_pred.json`
	- should be linear