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
              "A5V3ZMQI0PU3F", #
              "A17AF9I56P3RVV",
              "A1GV0UZU0T2ORS",
              "A22KRF782ELLB0",
              "A2YE7NBCK14VAQ",
              "A38DC3BG1ZCVZ2"
              ]

           
#recipients = ["A17AF9I56P3RVV",
#              "A1GV0UZU0T2ORS",
#              "A22KRF782ELLB0",
#             "A2YE7NBCK14VAQ",
#              "A38DC3BG1ZCVZ2"
#              ]





subject = "Final HITs available ($0.05 / ~30 sec. HIT) - Write what you would say in the given situation (camera shop scenario) x"

message = """
Dear Turkers,

Thank you so much for your work so far on our utterance data collection task!

We have made the final batch of our HITs (around 7000 HITs) available at a rate of $0.05 per task (~30 seconds per task). 

We will also pay a bonus of $37 to people who complete at least 4000 HITs.

Worker ID and HITs completed so far:
A1GV0UZU0T2ORS 3621
A34M93NJC830DP 2597
A2LV5432PV1S39 1598
A1O4TRWZO664V3 1420
A38DC3BG1ZCVZ2 1234
A3GNQDFPZALU92 1160
AKSJ3C5O3V9RB 1155
A2YE7NBCK14VAQ 853
A3UUH3632AI3ZX 168
A1S88VQY8G8CNC 127
A22KRF782ELLB0 125
AYJGJAIY0EXW 74
A1GGOZPHYU0OC0 51
A15L7DBIL7TYTW 19
A17AF9I56P3RVV 15
A5V3ZMQI0PU3F 5
A34QZDSTKZ3JO9 5
A2KG59JUICJLP0 2
A25D66AC4QUW2U 2
A9WGDBDXUNLOD 1
AKDXDUOMK3XVD 1
A1NGXQMOBCXDC3 1
A127F80LLYOQ22 1
AE861G0AY5RGT 1


Please check out our HITs if you are interested!


A note on the instructions:
It's not necessary to make up so much variation in phrasing in your responses (although some variation is good). 
Just pretend you are a shopkeeper working at a camera shop over several weeks - it's natural to have some variation, but you would probably usually speak in a consistent way.

Also, when writing out numbers, please write them as they appear in the camera information database instead of writing out the words.


Thank you!

Malcolm

"""




response = client.notify_workers(Subject=subject,
                                 MessageText=message,
                                 WorkerIds=recipients)

pass








