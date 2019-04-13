# from datetime import datetime
# start_time = datetime.now()
# print ("executing something...")
# now_time = datetime.now()
# print (now_time)

# import time
# start = time.clock()
# print ("executing something...")
# end = time.clock()
#
# print (str(end-start))

import datetime
starttime = datetime.datetime.now()
print ("Executing...")
print (9000/35/60)
endtime = datetime.datetime.now()
duration = (endtime - starttime).seconds
print ("Execution takes {} seconds".format(duration))