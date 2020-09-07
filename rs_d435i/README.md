# rs_d435i

### realsense driver installation
- offical url: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md

### realsense ros package installation
- offical url: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages

### My install note for d435i
- If you lazy to read the detail about how to install driver and package
  you can consider to take my install note for reference.
- I record my install note and put in src/rs_d435i/rs_d435i install guide
- Please install driver then install package.
- If you suffered from any problems, you can read or open issues in this repo.

### Environment setting
- Change the path of the following files: 
    * file1: get_rs_image/scripts/get_rs_image.Get_Image.py
    * file2: rs_d435i/test_rs_img/scripts/get_rs_module_img.py
- Change this: sys.path.insert(1, "/home/<your_pc_name>/.local/lib/python3.5/site-packages/")
    * Do change in file1 and file2
- Change this: sys.path.insert(2, "/home/<your_pc_name>/<your_workspace_name>/catkin_workspace/install/lib/python3/dist-packages")
    * Do change only in file2
- python may be different for different pc, please check your path
- cd src/rs_d435i/
- source create_catkin_ws.sh

### Display image from d435i by using ros
(terminal-1)
< cd to your workspace >
```  
. devel/setup.bash
roslaunch realsense2_camera rs_rgbd.launch
```

open new terminal
(terminal-2)
```
. /devel/setup.bash 
rosrun test_rs_img get_rs_module_img.py
```

NOTE: If you have a fault when you run bash scripts in terminal-2, you have to re-open a new terminal and re-execute the two bash scripts again.