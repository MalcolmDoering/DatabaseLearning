# -*- coding: utf-8 -*-
"""
Created on 2019 April 5

@author: MalcolmD

adapted from advancedSimulator7

modified to generate new databases and (re)initialize the simulator from a function
"""

import numpy as np
import copy
import csv
import time
import random


import tools
import attr


DEBUG_FLAG = True


databaseFilename = None
shopkeeperUtteranceFilename = None
customerUtteranceFilename = None

cameras = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]


features = [#"camera_name", 
            "camera_type", 
            "color", 
            "weight", 
            "preset_modes", 
            "effects", 
            "price" ,
            "resolution", 
            "optical_zoom", 
            "settings",
            "autofocus_points",
            "sensor_size",
            "ISO",
            "long_exposure"]


#additionalQuestionTopics = ["bulb_mode_what",
#                            "glamour_retouch_effects_what",
#                            "ISO_what",
#                           "megapixels_what",
#                            "optical_zoom_what",
#                            "preset_modes_what",
#                            "sensor_size_what",
#                            "artistic_effects_what",
#                            "exposure_what",
#                            "mirrorless_what"]
additionalQuestionTopics = []


fieldnames = ["TRIAL",
              "TURN_COUNT",
              
              "CURRENT_CAMERA_OF_CONVERSATION",
              "PREVIOUS_CAMERAS_OF_CONVERSATION",
              "PREVIOUS_FEATURES_OF_CONVERSATION",
              
              
              "CUSTOMER_ACTION",
              "CUSTOMER_LOCATION",
              "CUSTOMER_TOPIC",
              "CUSTOMER_SPEECH",
              "SPATIAL_STATE",
              "STATE_TARGET",
              
              
              "OUTPUT_SHOPKEEPER_ACTION",
              "OUTPUT_SHOPKEEPER_LOCATION",
              "SHOPKEEPER_TOPIC",
              "SHOPKEEPER_SPEECH",
              "OUTPUT_SPATIAL_STATE",
              "OUTPUT_STATE_TARGET",
              
              "SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"
              
              ]


def read_database_file(filename):
    database = {}
    
    for c in cameras:
        database[c] = {}
        
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            for att, val in list(row.items()):
                database[row["camera_ID"]][att] = val
                
    return database


def read_shopkeeper_utterance_file(filename, database):
    
    shopkeeperUtteranceMap = {}
    shkpUttToDbEntryRange = {}
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            act = row["ACTION"]
            top = row["TOPIC"]
            coc = row["CAMERA_OF_CONVERSATION"]
            utt = row["UTTERANCE"]
            
            if act not in shopkeeperUtteranceMap:
                shopkeeperUtteranceMap[act] = {}
            
            if top not in shopkeeperUtteranceMap[act]:
                shopkeeperUtteranceMap[act][top] = {}
                
            if coc not in shopkeeperUtteranceMap[act][top]:
                
                if coc == "CAMERA_X":
                    for c in cameras:
                        shopkeeperUtteranceMap[act][top][c] = []
                
                else:
                    shopkeeperUtteranceMap[act][top][coc] = []
            
            
            # add the utterance to the map, but replace "{}" with database information first        
            if utt != "" and utt != "#":
                
                if coc == "CAMERA_X":
                    for c in cameras:
                        
                        if top in database[c] and database[c][top] != "":
                            
                            dbEntryStartIndex = utt.find("{")
                            dbEntryEndIndex = dbEntryStartIndex + len(database[c][top])
                            
                            uttMod = utt.replace("{}", database[c][top])
                            shopkeeperUtteranceMap[act][top][c].append(uttMod)
                            
                            shkpUttToDbEntryRange[uttMod] = "{}~{}".format(dbEntryStartIndex, dbEntryEndIndex)
                            
                            
                        elif act == "S_INTRODUCES_CAMERA":
                            
                            dbEntryStartIndex = utt.find("{")
                            dbEntryEndIndex = dbEntryStartIndex + len(database[c]["camera_name"])
                            
                            uttMod = utt.replace("{}", database[c]["camera_name"])
                            shopkeeperUtteranceMap[act][top][c].append(uttMod)
                            
                            shkpUttToDbEntryRange[uttMod] = "{}~{}".format(dbEntryStartIndex, dbEntryEndIndex)
                            
                            
                        elif top not in database[c]:
                            shopkeeperUtteranceMap[act][top][c].append(utt)
                
                else:
                    shopkeeperUtteranceMap[act][top][coc].append(utt)
    
        for act in shopkeeperUtteranceMap:
            for top in shopkeeperUtteranceMap[act]:
                for coc in shopkeeperUtteranceMap[act][top]:
                    if len(shopkeeperUtteranceMap[act][top][coc]) == 0:
                        shopkeeperUtteranceMap[act][top][coc].append("")
    
    return shopkeeperUtteranceMap, shkpUttToDbEntryRange


def read_customer_utterance_file(filename):
    
    customerUtteranceMap = {}
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            act = row["ACTION"]
            top = row["TOPIC"]
            utt = row["UTTERANCE"]
            
            if act not in customerUtteranceMap:
                customerUtteranceMap[act] = {}
            
            if top not in customerUtteranceMap[act]:
                customerUtteranceMap[act][top] = []
            
            if utt != "":
                customerUtteranceMap[act][top].append(utt)
    
    for act in customerUtteranceMap:
        for top in customerUtteranceMap[act]:
            if len(customerUtteranceMap[act][top]) == 0:
                customerUtteranceMap[act][top].append("")
    
    return customerUtteranceMap


def initialize_simulator(databaseFilename, shopkeeperUtteranceFilename, customerUtteranceFilename):
    global database, shopkeeperUtteranceMap, shkpUttToDbEntryRange, customerUtteranceMap, shkpHasInfo
    
    database = read_database_file(databaseFilename)
    shopkeeperUtteranceMap, shkpUttToDbEntryRange = read_shopkeeper_utterance_file(shopkeeperUtteranceFilename, database)
    customerUtteranceMap = read_customer_utterance_file(customerUtteranceFilename)                            
        
    # which features of which cameras does the database contain information for?
    shkpHasInfo = {}
    
    for c in cameras:
        shkpHasInfo[c] = {}
        
        for f in features:
            
            if database[c][f] == "":    
                shkpHasInfo[c][f] = 0
            else:
                shkpHasInfo[c][f] = 1
    


class InteractionState(object):
    
    def __init__(self, trialId=-1):
        
        self.end = False
        
        self.trialId = str(trialId)
        self.turnCount = 0
        
        self.currentCameraOfConversation = None
        self.previousCamerasOfConversation = []
        self.previousFeaturesOfConversation = {"CAMERA_1":[], "CAMERA_2":[], "CAMERA_3":[]}
        
        self.customerAction = None
        self.customerTopic = None
        self.customerSpeech = None
        
        self.spatialState = None
        self.stateTarget = None
        
        self.customerLocation = None
        
        
        self.outputShopkeeperAction = None
        self.shopkeeperTopic = None
        self.shopkeeperSpeech = None
        
        self.outputSpatialState = None
        self.outputStateTarget = None
        
        self.outputShopkeeperLocation = None
    
        
    def to_dict(self):
        
        dictionary = {"CURRENT_CAMERA_OF_CONVERSATION":self.currentCameraOfConversation,
                      "PREVIOUS_CAMERAS_OF_CONVERSATION":self.previousCamerasOfConversation,
                      "PREVIOUS_FEATURES_OF_CONVERSATION":self.previousFeaturesOfConversation,
                      "TRIAL":self.trialId,
                      "TURN_COUNT":self.turnCount,
                      
                      "CUSTOMER_ACTION":self.customerAction,
                      "OUTPUT_SHOPKEEPER_ACTION":self.outputShopkeeperAction,
                      
                      "SHOPKEEPER_TOPIC":self.shopkeeperTopic,
                      "CUSTOMER_TOPIC":self.customerTopic,
                      
                      "CUSTOMER_SPEECH":self.customerSpeech,
                      "SHOPKEEPER_SPEECH":self.shopkeeperSpeech,
                      
                      "SPATIAL_STATE":self.spatialState,
                      "STATE_TARGET":self.stateTarget,
                      
                      "OUTPUT_SPATIAL_STATE":self.outputSpatialState,
                      "OUTPUT_STATE_TARGET":self.outputStateTarget,
                      
                      "CUSTOMER_LOCATION":self.customerLocation,
                      "OUTPUT_SHOPKEEPER_LOCATION":self.outputShopkeeperLocation
                      }
        
        if self.shopkeeperSpeech in shkpUttToDbEntryRange:
            dictionary["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"] = shkpUttToDbEntryRange[self.shopkeeperSpeech]
        else:
            dictionary["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"] = ""
        
        return dictionary




class CustomerAgent(object):
    
    def __init__(self):
        pass
    
    
    #
    # choose the next customer action and execute
    #
    def perform_action(self, prevIntState):
        
        currIntState = InteractionState(prevIntState.trialId)
        
        # copy the relevant information from the previous interaction state
        currIntState.currentCameraOfConversation = prevIntState.currentCameraOfConversation
        currIntState.previousCamerasOfConversation = copy.deepcopy(prevIntState.previousCamerasOfConversation)
        currIntState.previousFeaturesOfConversation = copy.deepcopy(prevIntState.previousFeaturesOfConversation)
        currIntState.trialId = prevIntState.trialId
            
        currIntState.turnCount = prevIntState.turnCount + 1
        
        
        if prevIntState.turnCount == 0:
            action = self.action_enters
        
        else:    
        
            if prevIntState.customerAction == "C_ENTERS":
                    action = np.random.choice([self.action_greets, self.action_walks_to_camera], p=[0.5, 0.5])
            
            elif prevIntState.outputShopkeeperAction == "S_GREETS":
                action = self.action_looking_for_a_camera
                
            elif prevIntState.outputShopkeeperAction == "S_INTRODUCES_CAMERA":
                action = np.random.choice([self.action_question_about_feature, self.action_silent_or_backchannel], p=[0.66, 0.34])
                
            elif prevIntState.outputShopkeeperAction == "S_LET_ME_KNOW_IF_YOU_NEED_HELP":
                action = self.action_walks_to_camera
            
            elif prevIntState.outputShopkeeperAction == "S_ANSWERS_QUESTION_ABOUT_FEATURE" or prevIntState.outputShopkeeperAction == "S_NOT_SURE":
                action = np.random.choice([self.action_question_about_feature, self.action_silent_or_backchannel, self.action_thank_you, self.action_anything_else_available], p=[0.50, 0.30, 0.10, 0.10])
                
            elif prevIntState.outputShopkeeperAction == "S_INTRODUCES_FEATURE":
                action = np.random.choice([self.action_silent_or_backchannel, self.action_question_about_feature, self.action_thank_you, self.action_anything_else_available], p=[0.35, 0.35, 0.20, 0.10])
                
            elif prevIntState.outputShopkeeperAction == "S_THATS_ALL_WE_HAVE_AVAILABLE":
                action = self.action_thank_you
                
            elif prevIntState.outputShopkeeperAction == "S_THANK_YOU":
                action = self.action_leaves
            
            elif prevIntState.customerAction == "C_WALKS_TO_CAMERA":
                action = np.random.choice([self.action_examines_camera, self.action_question_about_feature], p=[0.5, 0.5])
        
            elif prevIntState.customerAction == "C_EXAMINES_CAMERA":
                action = np.random.choice([self.action_examines_camera, self.action_walks_to_camera, self.action_question_about_feature], p=[0.25, 0.4, 0.35]) # "S_GREETS"
            
            elif prevIntState.customerAction == "C_THANK_YOU":
                action = np.random.choice([self.action_think_it_over, self.action_decides_to_buy], p=[0.5, 0.5]) # "S_THANK_YOU"
        
        
        currIntState = action(prevIntState, currIntState)
        
        
        #print currIntState.customerAction
        #print currIntState.customerTopic
        #print
        
        
        #
        # choose the customer speech and keywords
        #
        if currIntState.customerTopic == "NONE" or currIntState.customerTopic == "":
            top = ""
        else:
            top = currIntState.customerTopic
        
        currIntState.customerSpeech = np.random.choice(customerUtteranceMap[currIntState.customerAction][top])
        
        if DEBUG_FLAG:
            print(currIntState.customerAction, top)
        
        
        return currIntState
    
    
    def state_as_dict(self):
        customerState = {}
        return customerState
    
    
    #
    # C_ENTERS
    #
    def action_enters(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_ENTERS"
        
        currIntState.currentCameraOfConversation = "NONE"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.customerLocation = "DOOR"
        currIntState.shopkeeperLocation = "SERVICE_COUNTER"
        
        return currIntState
    
    
    #
    # C_GREETS
    #
    def action_greets(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_GREETS"
        
        currIntState.currentCameraOfConversation = "NONE"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.customerLocation = "MIDDLE"
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        return currIntState 
        
    #
    # C_WALKS_TO_CAMERA
    #
    def action_walks_to_camera(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_WALKS_TO_CAMERA"
        
        # choose a camera
        cam = None
        
        # which cameras haven't been talked about yet
        candidates = [c for c in cameras if c not in currIntState.previousCamerasOfConversation]
        
        # randomly choose a camera that has not been talked about yet and the customer is not currently at
        while cam == None or prevIntState.customerLocation == cam:
            
            if candidates != []:
                cam = np.random.choice(candidates)
            else:
                cam = np.random.choice(cameras)
        
        currIntState.currentCameraOfConversation = "NONE"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.customerLocation = cam
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
    
        return currIntState
    
    
    #
    # C_EXAMINES_CAMERA
    #
    def action_examines_camera(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_EXAMINES_CAMERA"
        
        currIntState.currentCameraOfConversation = "NONE"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        return currIntState
    
    
    
    #
    # C_SILENT_OR_BACKCHANNEL
    #
    def action_silent_or_backchannel(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_SILENT_OR_BACKCHANNEL"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
    
        return currIntState
    
    
    #
    # C_THANK_YOU
    #
    def action_thank_you(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_THANK_YOU"
        
        currIntState.currentCameraOfConversation = "NONE"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        return currIntState
    
    
    #
    # C_THINK_IT_OVER
    #
    def action_think_it_over(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_THINK_IT_OVER"
        
        currIntState.currentCameraOfConversation = "NONE"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        return currIntState
    
    
    #
    # C_DECIDE_TO_BUY
    #
    def action_decides_to_buy(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_DECIDE_TO_BUY"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        return currIntState
    
    
    #
    # C_LEAVES
    #
    def action_leaves(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_LEAVES"
        
        currIntState.currentCameraOfConversation = "NONE"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.customerLocation = "DOOR"
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        currIntState.outputCustomerLocation = "NONE"
        
        return currIntState
    
    
    #
    # C_LOOKING_FOR_A_CAMERA
    #
    def action_looking_for_a_camera(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_LOOKING_FOR_A_CAMERA"
        
        currIntState.currentCameraOfConversation = "NONE"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        return currIntState
    
    
    #
    # C_ANYTHING_ELSE_AVAILABLE
    #
    def action_anything_else_available(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_ANYTHING_ELSE_AVAILABLE"
        
        currIntState.currentCameraOfConversation = "NONE"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        return currIntState
    
    
    #
    # C_QUESTION_ABOUT_FEATURE
    #
    def action_question_about_feature(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_QUESTION_ABOUT_FEATURE"
        
        # if a camera is not already under discussion, start talking about the one the customer is at
        if currIntState.currentCameraOfConversation == "NONE":
            currIntState.currentCameraOfConversation = prevIntState.outputCustomerLocation
            
            if currIntState.currentCameraOfConversation not in cameras:
                print("WARNING: Customer is not currently located at a camera!")
        
        # choose a feature to ask about
        
        # which features have not yet been asked about
        candidateFeats = [x for x in features if x not in currIntState.previousFeaturesOfConversation[currIntState.currentCameraOfConversation]]
        candidateFeats += [x for x in additionalQuestionTopics if x not in currIntState.previousFeaturesOfConversation[currIntState.currentCameraOfConversation]]
        
        
        # choose a feature that has not already been talked about
        if len(candidateFeats) > 0:
            feat = np.random.choice(candidateFeats)
        
        # re-ask about any feature
        # TODO: it would be better to just go to a different camera or leave the store at this point
        else:
            feat = np.random.choice(features)
        
        
        currIntState.previousFeaturesOfConversation[currIntState.currentCameraOfConversation].append(feat)
        
        currIntState.customerTopic = feat
        currIntState.customerSpeech = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        return currIntState




class ShopkeeperAgent(object):
    
    def __init__(self):
        pass

    
    def state_as_dict(self):
        shopkeeperState = {}
        return shopkeeperState
    
    
    #
    # choose the next shopkeeper action and execute it
    #
    def perform_action(self, prevIntState, currIntState):
        
        if prevIntState.turnCount == 0:
            action = self.action_none
            
        else:
            if currIntState.customerAction == "C_GREETS":
                action = self.action_greets
                
            elif currIntState.customerAction == "C_LOOKING_FOR_A_CAMERA" or currIntState.customerAction == "C_ANYTHING_ELSE_AVAILABLE":
                action = self.action_introduces_camera
                
            elif currIntState.customerAction == "C_EXAMINES_CAMERA":
                action = np.random.choice([self.action_none, self.action_greets], p=[0.25, 0.75])
                
            elif currIntState.customerAction == "C_WALKS_TO_CAMERA":
                action = np.random.choice([self.action_none, self.action_greets], p=[0.25, 0.75])
            
            elif prevIntState.outputShopkeeperAction == "S_LET_ME_KNOW_IF_YOU_NEED_HELP":
                action = self.action_return_to_counter
                
            elif currIntState.customerAction == "C_QUESTION_ABOUT_FEATURE":
                action = self.action_answers_question_about_feature
                
            elif currIntState.customerAction == "C_SILENT_OR_BACKCHANNEL":
                if prevIntState.customerSpeech == "NONE" and currIntState.customerSpeech == "NONE":
                    action = self.action_anything_else_i_can_help_you_with
                else:
                    action = self.action_introduces_feature
            
            elif currIntState.customerAction == "C_THANK_YOU":
                action = self.action_thank_you
                
            elif currIntState.customerAction == "C_THINK_IT_OVER":
                action = self.action_thank_you
                
            elif currIntState.customerAction == "C_DECIDE_TO_BUY":
                action = self.action_thank_you
                
            elif currIntState.customerAction == "C_LEAVES":
                action = self.action_return_to_counter
        
        
        currIntState = action(currIntState)
        
        currIntState = self.update_previous_cameras_of_conversation(currIntState)
        

        #
        # choose the shopkeeper speech and keywords
        #
        if currIntState.shopkeeperTopic == "NONE" or currIntState.shopkeeperTopic == "" or currIntState.shopkeeperTopic == None:
            top = ""
        else:
            top = currIntState.shopkeeperTopic
        
        if currIntState.outputShopkeeperAction in ["S_NONE", 
                                                   "S_GREETS", 
                                                   "S_THANK_YOU", 
                                                   "S_RETURNS_TO_COUNTER", 
                                                   "S_LET_ME_KNOW_IF_YOU_NEED_HELP", 
                                                   "S_ANYTHING_ELSE_I_CAN_HELP_WITH", 
                                                   "S_GREET_RESPONSE", 
                                                   "S_NOT_SURE", 
                                                   "S_TRY_IT_OUT", 
                                                   "S_YOURE_WELCOME"]:
            coc = ""
            
        else:
            coc = currIntState.currentCameraOfConversation
        
        if DEBUG_FLAG:
            print(currIntState.outputShopkeeperAction, top, coc)
        
        
        try:
            currIntState.shopkeeperSpeech = np.random.choice(shopkeeperUtteranceMap[currIntState.outputShopkeeperAction][top][coc])
        except Exception as e:
            print(str(e))
            pass
        
        return currIntState
    
    
    #
    # update the history of which cameras have been talked about
    #
    def update_previous_cameras_of_conversation(self, currIntState):
        if currIntState.currentCameraOfConversation != "NONE" and currIntState.currentCameraOfConversation != "":
            
            if len(currIntState.previousCamerasOfConversation) == 0 or (len(currIntState.previousCamerasOfConversation) > 0 and currIntState.previousCamerasOfConversation[-1] != currIntState.currentCameraOfConversation):
                currIntState.previousCamerasOfConversation.append(currIntState.currentCameraOfConversation)
        
        return currIntState
    
    
    #
    # S_NONE
    #
    def action_none(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_NONE"
        
        currIntState.currentCameraOfConversation = "NONE"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = currIntState.shopkeeperLocation
        
        return currIntState
    
    
    #
    # S_GREETS
    #
    def action_greets(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_GREETS"
        
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.currentCameraOfConversation = "NONE"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_THANK_YOU
    #
    def action_thank_you(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_THANK_YOU"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = currIntState.shopkeeperLocation
        
        return currIntState
    
    
    #
    # S_RETURNS_TO_COUNTER
    #
    def action_return_to_counter(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_RETURNS_TO_COUNTER"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = "SERVICE_COUNTER"
        
        return currIntState
    
    
    #
    # S_LET_ME_KNOW_IF_YOU_NEED_HELP
    #
    def action_let_me_know_if_you_need_help(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_LET_ME_KNOW_IF_YOU_NEED_HELP"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = "SERVICE_COUNTER"
        
        return currIntState
    
    
    #
    # S_ANYTHING_ELSE_I_CAN_HELP_WITH
    #
    def action_anything_else_i_can_help_you_with(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_ANYTHING_ELSE_I_CAN_HELP_WITH"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_GOODBYE
    #
    def action_goodbye(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_GOODBYE"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = "SERVICE_COUNTER"
        
        return currIntState
    
    
    #
    # S_GREET_RESPONSE
    #
    def action_greet_response(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_GREET_RESPONSE"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_NOT_SURE
    #
    def action_not_sure(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_NOT_SURE"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_TRY_IT_OUT
    #
    def action_try_it_out(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_TRY_IT_OUT"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_YOURE_WELCOME
    #
    def action_youre_welcome(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_YOURE_WELCOME"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_ANSWERS_QUESTION_ABOUT_FEATURE
    #
    def action_answers_question_about_feature(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_ANSWERS_QUESTION_ABOUT_FEATURE"
        
        top = currIntState.customerTopic
        
        if top not in additionalQuestionTopics and shkpHasInfo[currIntState.currentCameraOfConversation][top] == 0:
            top = ""
            currIntState.outputShopkeeperAction = "S_NOT_SURE"
        
        
        currIntState.shopkeeperTopic = top
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_INTRODUCES_FEATURE
    #
    def action_introduces_feature(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_INTRODUCES_FEATURE"
        
        currFeat = None
        
        currCamFeats = [x for x in features if shkpHasInfo[currIntState.currentCameraOfConversation][x] == 1]
        
        
        # features that the current camera have that have not been talked about yet
        candidateFeats = [x for x in currCamFeats if x not in currIntState.previousFeaturesOfConversation[currIntState.currentCameraOfConversation]]
        
        # introduce a random feature that hasn't been talked about yet
        if len(candidateFeats) > 0:
            currFeat = np.random.choice(candidateFeats)
        
        # if all features have already been introduced, restate a random feature
        else:
            currFeat = np.random.choice(currCamFeats)
        
        
        currIntState.shopkeeperTopic = currFeat
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        currIntState.previousFeaturesOfConversation[currIntState.currentCameraOfConversation].append(currIntState.shopkeeperTopic)
        
        return currIntState
        
    
    #
    # S_INTRODUCES_CAMERA
    #
    def action_introduces_camera(self, currIntState):
        
        # choose a camera
        # which cameras haven't been talked about yet
        candidateCams = [c for c in cameras if c not in currIntState.previousCamerasOfConversation]
        
        
        if len(candidateCams) > 0:
            currCamera = np.random.choice(candidateCams)
            
            currIntState.outputShopkeeperAction = "S_INTRODUCES_CAMERA"
            currIntState.currentCameraOfConversation = currCamera
            
            currIntState.shopkeeperTopic = ""
            
            currIntState.shopkeeperSpeech = ""
            currIntState.outputCustomerLocation = currCamera
            currIntState.outputShopkeeperLocation = currCamera
            
        
        elif currIntState.customerAction == "C_ANYTHING_ELSE_AVAILABLE":
            currIntState = self.action_thats_all_we_have_available(currIntState)
            
        else:        
            currIntState = self.action_anything_else_i_can_help_you_with(currIntState)
        
        
        return currIntState


    #
    # S_THATS_ALL_WE_HAVE_AVAILABLE
    #
    def action_thats_all_we_have_available(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_THATS_ALL_WE_HAVE_AVAILABLE"
        
        currIntState.currentCameraOfConversation = ""
        currIntState.shopkeeperTopic = ""
        currIntState.shopkeeperSpeech = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState


def set_spatial_formation(prevIntState, currIntState):
    
    
    # for before the shopkeeper acts
    if prevIntState.turnCount == 0:
        # the begining of the interaction
        currIntState.spatialState = "WAITING"
        currIntState.stateTarget = "NONE"
        
    else:
        currIntState.spatialState = prevIntState.outputSpatialState
        currIntState.stateTarget = prevIntState.stateTarget
    
    
    # for after the shopkeeper acts
    if currIntState.customerLocation in cameras and currIntState.outputShopkeeperLocation == currIntState.customerLocation:
        currIntState.outputSpatialState = "PRESENT_X"
        currIntState.outputStateTarget = currIntState.outputShopkeeperLocation
    
    elif currIntState.customerLocation not in cameras and currIntState.outputShopkeeperLocation == currIntState.customerLocation:
        currIntState.outputSpatialState = "FACE_TO_FACE"
        currIntState.outputStateTarget = "NONE"
    
    elif currIntState.outputShopkeeperLocation == "SERVICE_COUNTER":
        currIntState.outputSpatialState = "WAITING"
        currIntState.outputStateTarget = "NONE"
    
    if currIntState.outputShopkeeperAction == "S_INTRODUCES_CAMERA":
        currIntState.outputSpatialState = "PRESENT_X"
        currIntState.outputStateTarget = currIntState.outputShopkeeperLocation
    
    if currIntState.outputShopkeeperAction == "S_GREETS":
        currIntState.outputSpatialState = "FACE_TO_FACE"
        currIntState.outputStateTarget = "NONE"
    
    if currIntState.outputShopkeeperAction == "S_THANK_YOU":
        currIntState.outputSpatialState = "FACE_TO_FACE"
        currIntState.outputStateTarget = "NONE"
    
    

    
    return currIntState


def simulate_interaction(trialId, seed):
    #print "trial id:", trialId, "seed:", seed
    
    np.random.seed(seed)
    interaction = []    
    
    prevIntState = InteractionState(trialId)
    currIntState = None
    
    custAgent = CustomerAgent()
    shkpAgent = ShopkeeperAgent()
    
    while True:
        
        currIntState = custAgent.perform_action(prevIntState)
        currIntState = shkpAgent.perform_action(prevIntState, currIntState)
        
        currIntState = set_spatial_formation(prevIntState, currIntState)
        
        #print currIntState.turnCount
        
        interaction.append(currIntState.to_dict())
        
        
        if currIntState.outputShopkeeperAction == "S_INTRODUCES_CAMERA" and currIntState.outputShopkeeperLocation == "MIDDLE":
            pass
        
        
        if currIntState.customerAction == "C_LEAVES":
            break
        
        prevIntState = copy.deepcopy(currIntState)
        
    
    return interaction



def simulate_n_interactions(n, flatten=True, startSeed=0):
    
    interactions = []
    
    for i in range(n):
        if flatten:
            interactions += simulate_interaction(i, startSeed+i)
        else:
            interactions.append(simulate_interaction(i, startSeed+i))
        
        if DEBUG_FLAG:
            print()
    
    return interactions



def generate_databases(originalDatabaseFilename, numDatabasesToGenerate, randomSeed=0):
    
    random.seed(randomSeed)
    
    originalDatabase = []
    newDatabaseFilenames = []
    
    
    dbId = databaseFilename.split("_")[-1].split(".")[0]
    today = time.strftime("%Y-%m-%d")
    
    
    with open(originalDatabaseFilename) as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            originalDatabase.append(row)
    
    
    for i in range(numDatabasesToGenerate):
        
        # generate new random prices for the cameras in the original database
        newDatabase = copy.deepcopy(originalDatabase)
        newDatabaseFilename = sessionDir + "/{}_database_{}-{:02d}.csv".format(today, dbId, i)
        newDatabaseFilenames.append(newDatabaseFilename)
        
        for j in range(len(newDatabase)):
            newDatabase[j]["price"] = "$" + str(random.randrange(50, 5000))
        
        # save
        with open(newDatabaseFilename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(newDatabase)
    
    return newDatabaseFilenames



def generate_databases_2(numDatabasesToGenerate, randomSeed=0):
    """randomly generate all attributes, not just price"""
    
    fieldnames = ["camera_ID", "camera_name", "camera_type", "color", "weight", "preset_modes", "effects", "price", "resolution", "optical_zoom", "settings", "autofocus_points", "sensor_size", "ISO", "long_exposure"]
    
    newDatabaseFilenames = []
    
    for i in range(numDatabasesToGenerate):

        dbId = "generated"
        today = time.strftime("%Y-%m-%d")
        
        newDatabaseFilename = sessionDir + "/{}_database_{}-{:02d}.csv".format(today, dbId, i)
        newDatabaseFilenames.append(newDatabaseFilename)
        
        newDatabase = []
        
        # generate attributes for 3 cameras
        for r in range(3):
            row = {}
            
            row["camera_ID"] = "CAMERA_{}".format(r+1)
            row["camera_name"] = generate_camera_name()
            row["camera_type"] = generate_camera_type()
            row["color"] = generate_color()
            row["weight"] = generate_weight()
            row["preset_modes"] = generate_preset_modes()
            row["effects"] = generate_effects()
            row["price"] = generate_price()
            row["resolution"] = generate_resolution()
            row["optical_zoom"] = generate_optical_zoom()
            row["settings"] = generate_settings()
            row["autofocus_points"] = generate_autofocus_points()
            row["sensor_size"] = generate_sensor_size()
            row["ISO"] = generate_ISO()
            row["long_exposure"] = generate_long_exposure()
            
            newDatabase.append(row)
        
        # save
        with open(newDatabaseFilename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(newDatabase)
    
    return newDatabaseFilenames



def generate_camera_name():
    
    component1 = ["Nikon", "Sony", "Canon", "Fujifilm", "Panasonic"]
    
    component2 = ["Coolpix", "Alpha", "EOS 5D", "PowerShot ELPH", "X-T20", "LUMIX GH5s C4K", "Lumix", "X100T"]
    
    component3 = ["S2800", "a6000", "Mark III", "360 HS", "ILC Camera", "DMC-ZS3", "A900"]
    
    attr = "{} {} {}".format(np.random.choice(component1), np.random.choice(component2), np.random.choice(component3))
    
    return attr


def generate_camera_type():
    
    types = ["point and shoot", "mirrorless", "DSLR", "digital single lens mirrorless", "compact digital camera", "premium compact digital camera"]
    
    attr = random.choice(types)
    
    return attr


def generate_color():
    colors = ["black", "white", "silver", "red", "purple", "pink", "blue"]
    numColors = random.randrange(1, len(colors)+1)
    
    selectedColors = random.sample(colors, numColors)
    
    if numColors == 1:
        attr = selectedColors[0]
    elif numColors == 2:
        attr = " or ".join(selectedColors)
    else:
        attr = ", ".join(selectedColors[:-1])
        attr += ", or " + selectedColors[-1]
    
    return attr


def generate_weight():
    weight = random.randrange(12, 100) * 10
    attr = "{} grams".format(weight)
    
    if random.random() < 0.3:
        attr += " (only body)"
    
    return attr


def generate_preset_modes():
    
    if random.random() < 0.33:
        attr = ""
    
    elif random.random() < 0.11:
        attr = "various preset modes"
    
    else:
        numPresetModes = random.randrange(9, 30)
        
        examples = ["beach", "snow", "night", "portrait", "fireworks", "low light", "soft skin", "sports", "beach", "backlighting", "close-up"]
        
        selectedExamples = random.sample(examples, 3)
        selectedExamples = ", ".join(selectedExamples)
        
        modeType = np.random.choice(["preset modes", "exposure modes"], p=[0.8, 0.2])
        
        attr = "{} {} (e.g. {})".format(numPresetModes, modeType, selectedExamples)
        
        if random.random() < 0.2:
            attr = "more  than " + attr
    
    return attr


def generate_effects():
    
    if random.random() < 0.33:
        attr = ""
    
    else:
        modeType = np.random.choice(["glamor retouch effects", "artistic effect modes", "photo effects", "advanced filters", "creative mode settings", "advanced filters"])
        
        examples = ["skin softening", "small face", "big eyes", "toy camera", "soft focus", "vivid", "lighter or darker skin tones", "sepia", "miniature", "retro"]
        selectedExamples = random.sample(examples, 3)
        selectedExamples = ", ".join(selectedExamples)
        
        numModes = random.randrange(8, 25)
        
        attr = "{} {} ({}, etc.)".format(numModes, modeType, selectedExamples)
        
    return attr


def generate_price():
    attr = "$" + str(random.randrange(50, 5000))
    return attr


def generate_resolution():
    
    if random.random() < 0.22:
        attr = ""
    
    else:
        resolution = random.randrange(100, 250) / 10
        
        attr = "{} megapixels".format(resolution)
    
    return attr


def generate_optical_zoom():
    
    if random.random() < 0.44:
        attr = ""

    else:    
        if random.random() < 0.2:
            attr = "no optical zoom"
        else:
            zoom = random.randrange(5, 35)
            attr = "{}x optical zoom".format(zoom)
        
    return attr


def generate_settings():
    
    attr = np.random.choice(["full manual control", "limited manual settings", "manual and automatic settings", "manual control and various automatic settings (video modeburst shooting, etc.)"], 
                            p=[0.22, 0.33, 0.34, 0.11])
    
    return attr


def generate_autofocus_points():
    
    if random.random() < 0.22:
        attr = ""
    
    else:
        numAfPoints = random.randrange(10, 230)    
        
        attr = "{} autofocus points".format(numAfPoints)
    
    return attr


def generate_sensor_size():
    
    if random.random() < 0.11:
        attr = ""
    
    else:
        sensorOptions = ["1/2.3 inch CMOS sensor",
                         "23.6mm x 15.6mm (APS-C) X-Trans CMOS II with primary color filter",
                         "23.5mm x 15.6mm [APS-C] X-Trans CMOS III with primary color filter",
                         "17.3 x 13.0 mm (in 4:3 aspect ratio)",
                         "APS-C sensor (crop sensor)",
                         "full frame",
                         "1/2.33 inch sensor"]
        
        attr = np.random.choice(sensorOptions)
    
    return attr


def generate_ISO():
    
    if random.random() < 0.11:
        attr = ""
    
    else:
        options = ["3200", "6400", "12800", "51200"]
        
        attr = "up to {} ISO".format(np.random.choice(options))
        
        if random.random() < 0.25:
            attr += " with little noise"
    
    return attr


def generate_long_exposure():
    
    if random.random() < 0.11:
        attr = ""
    
    else:
        duration = np.random.choice([15, 25, 30, 60], p=[0.125, 0.125, 0.375, 0.375])
        
        units = np.random.choice(["seconds", "minutes"])
        
        if units == "seconds":
            
            if random.random() < .2:
                extraMode = ", with bulb mode expose as long as you want"
            elif random.random() < .2:
                extraMode = " with Long Shutter Mode"
            elif random.random() < 0.2:
                extraMode = " with starry sky mode"
            else:
                extraMode = ""
        
        else:
            extraMode = " with bulb mode"
        
        attr = "{} {}".format(duration, units)
        
        if random.random() < 0.25:
            attr = "up to " + attr
        else:
            attr += " max"
        
        attr += extraMode
    
    return attr


def run_simulator(sessionDir, databaseFilename, shopkeeperUtteranceFilename, customerUtteranceFilename, randomSeed):
    
    numInteractions = 1000
    
    initialize_simulator(databaseFilename, shopkeeperUtteranceFilename, customerUtteranceFilename)
    
    dbId = databaseFilename.split("_")[-1].split(".")[0]
    
    #generate_shopkeeper_utterance_file()
    
    #global shkpActToUttMap
    #shkpActToUttMap = read_shopkeeper_utterance_file(sessionDir + "/shopkeeper_utterance_data.csv")
    
    
    interactions = simulate_n_interactions(numInteractions, flatten=False, startSeed=randomSeed)
    
    #
    # save to file
    #
    today = time.strftime("%Y-%m-%d")
    
    
    with open(sessionDir+"/{}_simulated_data_{}_database_{}.csv".format(today, numInteractions, dbId), "w", newline="") as csvfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for interaction in interactions:
            for row in interaction:
                writer.writerow(row)
                
    
    
    print("finished random seed", randomSeed)
    
    interactionLens = [len(i) for i in interactions]
    
    print(interactionLens)
    print("ave interaction len is", round(np.mean(interactionLens)), "s.d.", round(np.std(interactionLens)))
    
    
    
    

if __name__ == "__main__":
    
    sessionDir = tools.create_session_dir("advancedSimulator8")
    
    #global DEBUG_FLAG
    DEBUG_FLAG = False
    
    
    print("started")
    
    shopkeeperUtteranceFilename = tools.modelDir+"2019-04-04_shopkeeper_utterance_data.csv"
    #customerUtteranceFilename = tools.modelDir+"2019-04-04_customer_utterance_data.csv"
    customerUtteranceFilename = tools.modelDir+"2019-07-03_customer_utterance_data.csv"
    
    """
    # simulate interactions with databases where only the prices change
    databaseFilename = tools.dataDir+"database_0.csv"
    newDatabaseFilenames = generate_databases(databaseFilename, 11)
    
    for i in range(len(newDatabaseFilenames)):
        run_simulator(sessionDir, newDatabaseFilenames[i], shopkeeperUtteranceFilename, customerUtteranceFilename, randomSeed=i)
    """ 
    
    # simulate interactions where all the database fileds are generated randomly
    newDatabaseFilenames = generate_databases_2(101)
    
    for i in range(len(newDatabaseFilenames)):
        run_simulator(sessionDir, newDatabaseFilenames[i], shopkeeperUtteranceFilename, customerUtteranceFilename, randomSeed=i)
    
    
    """
    # simulate interactions with the handmade databases
    dbFilenames = [tools.dataDir+"/handmade_0/handmade_database_0-00.csv",
                   tools.dataDir+"/handmade_0/handmade_database_0-01.csv",
                   tools.dataDir+"/handmade_0/handmade_database_0-02.csv"]
    
    for i in range(len(dbFilenames)):
        run_simulator(sessionDir, dbFilenames[i], shopkeeperUtteranceFilename, customerUtteranceFilename, randomSeed=i)
    """
    
    