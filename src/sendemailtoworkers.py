'''
Created on Sep 27, 2019

@author: robovie
'''


#import mturk
import boto3



useSandbox = False


if useSandbox:
    endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
else:
    endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'




client = boto3.client('mturk')


#                      aws_access_key_id=ACCESS_KEY,
#                      aws_secret_access_key=SECRET_KEY,
#                      region_name=REGION


#recipients = ["A351QT25979V5G"]

recipients = ["A34M93NJC830DP", 
              "A2LV5432PV1S39", #
              "A3GNQDFPZALU92", #
              "AKSJ3C5O3V9RB", #
              "A1O4TRWZO664V3", #
              "A15L7DBIL7TYTW", #
              "A1GGOZPHYU0OC0", #
              "A3UUH3632AI3ZX", #
              "A5V3ZMQI0PU3F" #
              ]





subject = "More HITs available ($0.06 / ~30 sec. HIT) - Write what you would say in the given situation (camera shop scenario)"

message = """
Hello,

Thank you for your previous work on our task "Write what you would say in the given situation (camera shop scenario)."

We like the quality of your work and have made more HITs available at a higher pay rate ($0.06 per task, 30~40 seconds per task). 

Please check them out if you are interested!


A note on the instructions: 
It's not necessary to make up so much variation in phrasing in your responses (although some variation is good). 
Just pretend you are a shopkeeper working at a camera shop over several weeks - it's natural to have some variation, but you would probably usually speak in a consistent way.


Thank you!

Malcolm

"""




response = client.notify_workers(Subject=subject,
                                 MessageText=message,
                                 WorkerIds=recipients)

pass








