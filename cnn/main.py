import read_files as rf
import sys


print (sys.argv[1])

if len(sys.argv) < 1:
    print 'Please put the arguments'
    exit
else:
    if sys.argv[1] == "--help":
        print 'Please use the the follow sintaxe: script <path_to_files>'
        exit


read_file = rf.InputFileReader(sys.argv[1])
activit_data, weather_data = read_file.get_data()

print(activit_data)

print(weather_data)
