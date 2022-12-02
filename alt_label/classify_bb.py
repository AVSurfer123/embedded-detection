
#################### Adjustable Parameters ########################################################

#Characteristics need to be tuned based on sampled data, situationally

#speed characteristics come in list of [lower_bound, upper_bound]
#speed ~= m/s
label_keys = {
    0 : 2, #bike
    1 : 1, #ped
    2 : 41,#skate/scooter  #skateboard label, needs to learn scooters
    3 : 3, #car
    4 : 99, #golfcart #92 for new label or 3 for car
    5:  6  #bus

}
speed_chars = {
    "bike" :	[5, 10],
    "ped":	[.01, 5],
    "ska_scoot":[8, 15],
    "car":	[10, 20],
    "golf_cart":[8, 15],
    "bus":	[10, 15]
}
# alpha = height/width of bounding box
alpha_vals  = {
    "bike" :    .75,
    "ped":      2.5,
    "ska_scoot":3,
    "car":      .5,
    "golf_cart":1.5,
    "bus":      .2
}
#scale factor to adjust the importance of how different speed effects output probability
scale = 4
o_scale = scale / (scale + 1)
	
#####################################################################################################

def classify_bb(speed, alpha):
    #need function to classify whether a speed and scale ratio match a certain label group
    #and if not, find the closest match
    diffs = []
    total_diffs = 0
    count = 0
    for label in speed_chars:
        #compute difference in speed to characteristic range
        if speed > speed_chars[label][1]:
            diffs.append(1/scale * (speed - speed_chars[label][1]))
        elif speed < speed_chars[label][0]:
            diffs.append(1/scale * (speed_chars[label][0] - speed))
        else:
            diffs.append(0)
        # Undo Comments if trying to tune parameters!
	#print(label)
        #print("speed diff")
        #print(diffs[count])
        #add variation in alpha ratio
        diffs[count] = diffs[count] + (abs(alpha - alpha_vals[label]))*o_scale
        #print("alpha diff")
        #print(abs(alpha - alpha_vals[label]))#print(label)
        
        count = count + 1
        
    #keep running total of all diffs for probability, going to try different method
    #total_diffs  = total_diffs + diffs[label]
    print(diffs)
    max_diff =  min(diffs)
    max_ind = diffs.index(max_diff)
    return label_keys[max_ind]



        

