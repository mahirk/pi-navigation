import qwiic
import time

print("VL53L1X Qwiic Test\n")
ToF = qwiic.QwiicVL53L1X()
if ToF.sensor_init() == None:  # Begin returns 0 on a good init
    print("Sensor online!\n")
ToF.start_ranging()
while True:
    try:
        # Write configuration bytes to initiate measurement
        time.sleep(0.005)
        distance = (
            ToF.get_distance()
        )  # Get the result of the measurement from the sensor
        time.sleep(0.005)

        distanceInches = distance / (25.4 * 12.0)
        distanceFeet = distance / 12.0

        print("Distance(mm): %s Distance(ft): %s" % (distance, distanceInches))

    except Exception as e:
        print(e)
        ToF.stop_ranging()
