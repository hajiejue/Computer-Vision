We have used Census algorithm to calculate the disparity map. 
To run the programm you need to set the path of the image in the "challenge.m".

File introduction:
        challenge.m:             The main program, select the image path to get all the desired results.
        disparity_map.m:      Set different parameters for different pictures according to the set path, and return to the disparity map D and Euclidean transformation R,T.
        start_gui.m:              Open the interface corresponding to the program.
        verify_dmap.m:         Receive the calculated depth map and its corresponding ground truth to calculate the PSNR.    


Program running method:
    There are two ways to run the entire program.
    1. Run the file "start_gui.m" to open the interface. Select the catalog of desired image and the ground truth file in the drop down menu on the left and click the "Generate" button. 
        The program will automatically select the appropriate parameters, and will display the disparity map, Euclidean transform, PSNR and running time to the interface.
    2. You can also run the "Challenge.m" file directly. Note that you need to manually select the path of desired image and its ground truth.




Parameters and required time for each disparity map (with CPU intel core i7-8550U):

		window size for                searching range for 		required time
	               census calculation	        corresponding points

playground:                    7X7                                        20                                            7.93s

motorcycle:	               25X25	                                  250		                  circa 29min

sword:                           25X25                                     370                                       circa 37min                        

terrace:                           11X11                                        16                                            8.06s
