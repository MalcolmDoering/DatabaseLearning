# -*- coding: utf-8 -*-
"""
Created on 2019 April 4

@author: MalcolmD

adapted from advancedSimulator5 from the memory project.
modified for database learning project.
"""

import numpy as np
import copy
import csv


import tools

DEBUG = True

cameras = ["SONY", "CANON", "NIKON"]


featureValues = {"NIKON":{"accessories":0,
                                "artistic_effects":0,
                                "autofocus_points":0,
                                "black":1,
                                "bulb_mode":0,
                                "cheap":1,
                                "color":1,
                                "crop_sensor":0,
                                "depth_of_field":0,
                                "DSLR":0,
                                "ease_of_use":1,
                                "effects":1,
                                "expensive":0,
                                "exposure":0,
                                "full_frame":0,
                                "glamour_retouch_effects":1,
                                "high_performance":0,
                                "image_quality":1,
                                "ISO":0,
                                "lenses":0,
                                "light":1,
                                "long_exposure":0,
                                "low_light_performance":1,
                                "manual_settings":0,
                                "megapixels":1,
                                "midprice":0,
                                "mirrorless":0,
                                "movement":0,
                                "optical_zoom":1,
                                "pink":1,
                                "point_and_shoot":1,
                                "preset_modes":1,
                                "price":0,
                                "purple":1,
                                "red":1,
                                "sensor_size":0,
                                "shutter_speed":0,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":0,
                                "white":0},
                "SONY":{"accessories":0,
                                "artistic_effects":1,
                                "autofocus_points":1,
                                "black":1,
                                "bulb_mode":0,
                                "cheap":0,
                                "color":1,
                                "crop_sensor":1,
                                "depth_of_field":0,
                                "DSLR":0,
                                "ease_of_use":1,
                                "effects":1,
                                "expensive":0,
                                "exposure":1,
                                "full_frame":0,
                                "glamour_retouch_effects":0,
                                "high_performance":1,
                                "image_quality":1,
                                "ISO":1,
                                "lenses":0,
                                "light":1,
                                "long_exposure":1,
                                "low_light_performance":1,
                                "manual_settings":1,
                                "megapixels":0,
                                "midprice":1,
                                "mirrorless":1,
                                "movement":1,
                                "optical_zoom":0,
                                "pink":0,
                                "point_and_shoot":0,
                                "preset_modes":1,
                                "price":0,
                                "purple":0,
                                "red":0,
                                "sensor_size":0,
                                "shutter_speed":0,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":0,
                                "white":1},
                "CANON":{"accessories":0,
                                "artistic_effects":0,
                                "autofocus_points":1,
                                "black":1,
                                "bulb_mode":1,
                                "cheap":0,
                                "color":0,
                                "crop_sensor":0,
                                "depth_of_field":0,
                                "DSLR":1,
                                "ease_of_use":0,
                                "effects":0,
                                "expensive":1,
                                "exposure":1,
                                "full_frame":1,
                                "glamour_retouch_effects":0,
                                "high_performance":1,
                                "image_quality":1,
                                "ISO":1,
                                "lenses":1,
                                "light":0,
                                "long_exposure":1,
                                "low_light_performance":1,
                                "manual_settings":1,
                                "megapixels":0,
                                "midprice":0,
                                "mirrorless":0,
                                "movement":1,
                                "optical_zoom":0,
                                "pink":0,
                                "point_and_shoot":0,
                                "preset_modes":0,
                                "price":0,
                                "purple":0,
                                "red":0,
                                "sensor_size":0,
                                "shutter_speed":0,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":0,
                                "white":0}
                }

featuresForPicType = {"family_and_friends":{"accessories":0,
                                "artistic_effects":1,
                                "autofocus_points":0,
                                "black":0,
                                "bulb_mode":0,
                                "cheap":0,
                                "color":0,
                                "crop_sensor":0,
                                "depth_of_field":0,
                                "DSLR":0,
                                "ease_of_use":1,
                                "effects":1,
                                "expensive":0,
                                "exposure":0,
                                "full_frame":0,
                                "glamour_retouch_effects":1,
                                "high_performance":0,
                                "image_quality":1,
                                "ISO":0,
                                "lenses":0,
                                "light":0,
                                "long_exposure":0,
                                "low_light_performance":0,
                                "manual_settings":0,
                                "megapixels":0,
                                "midprice":0,
                                "mirrorless":0,
                                "movement":0,
                                "optical_zoom":0,
                                "pink":0,
                                "point_and_shoot":1,
                                "preset_modes":1,
                                "price":0,
                                "purple":0,
                                "red":0,
                                "sensor_size":0,
                                "shutter_speed":0,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":0,
                                "white":0},
                      "night":{"accessories":0,
                                "artistic_effects":0,
                                "autofocus_points":0,
                                "black":0,
                                "bulb_mode":1,
                                "cheap":0,
                                "color":0,
                                "crop_sensor":0,
                                "depth_of_field":0,
                                "DSLR":0,
                                "ease_of_use":0,
                                "effects":0,
                                "expensive":0,
                                "exposure":1,
                                "full_frame":0,
                                "glamour_retouch_effects":0,
                                "high_performance":1,
                                "image_quality":1,
                                "ISO":1,
                                "lenses":0,
                                "light":0,
                                "long_exposure":1,
                                "low_light_performance":1,
                                "manual_settings":1,
                                "megapixels":0,
                                "midprice":0,
                                "mirrorless":0,
                                "movement":0,
                                "optical_zoom":0,
                                "pink":0,
                                "point_and_shoot":0,
                                "preset_modes":1,
                                "price":0,
                                "purple":0,
                                "red":0,
                                "sensor_size":0,
                                "shutter_speed":0,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":0,
                                "white":0},
                      "travel":{"accessories":0,
                                "artistic_effects":0,
                                "autofocus_points":0,
                                "black":0,
                                "bulb_mode":0,
                                "cheap":0,
                                "color":0,
                                "crop_sensor":0,
                                "depth_of_field":0,
                                "DSLR":0,
                                "ease_of_use":1,
                                "effects":1,
                                "expensive":0,
                                "exposure":0,
                                "full_frame":0,
                                "glamour_retouch_effects":0,
                                "high_performance":0,
                                "image_quality":1,
                                "ISO":0,
                                "lenses":0,
                                "light":1,
                                "long_exposure":0,
                                "low_light_performance":0,
                                "manual_settings":0,
                                "megapixels":0,
                                "midprice":0,
                                "mirrorless":0,
                                "movement":0,
                                "optical_zoom":0,
                                "pink":0,
                                "point_and_shoot":0,
                                "preset_modes":1,
                                "price":0,
                                "purple":0,
                                "red":0,
                                "sensor_size":0,
                                "shutter_speed":0,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":0,
                                "white":0},
                      "outdoor":{"accessories":1,
                                "artistic_effects":0,
                                "autofocus_points":1,
                                "black":0,
                                "bulb_mode":0,
                                "cheap":0,
                                "color":0,
                                "crop_sensor":0,
                                "depth_of_field":0,
                                "DSLR":0,
                                "ease_of_use":0,
                                "effects":0,
                                "expensive":0,
                                "exposure":0,
                                "full_frame":0,
                                "glamour_retouch_effects":0,
                                "high_performance":1,
                                "image_quality":1,
                                "ISO":0,
                                "lenses":0,
                                "light":0,
                                "long_exposure":0,
                                "low_light_performance":0,
                                "manual_settings":1,
                                "megapixels":0,
                                "midprice":0,
                                "mirrorless":0,
                                "movement":0,
                                "optical_zoom":1,
                                "pink":0,
                                "point_and_shoot":0,
                                "preset_modes":1,
                                "price":0,
                                "purple":0,
                                "red":0,
                                "sensor_size":0,
                                "shutter_speed":0,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":0,
                                "white":0},
                      "sports":{"accessories":0,
                                "artistic_effects":0,
                                "autofocus_points":1,
                                "black":0,
                                "bulb_mode":0,
                                "cheap":0,
                                "color":0,
                                "crop_sensor":0,
                                "depth_of_field":0,
                                "DSLR":0,
                                "ease_of_use":0,
                                "effects":0,
                                "expensive":0,
                                "exposure":0,
                                "full_frame":0,
                                "glamour_retouch_effects":0,
                                "high_performance":0,
                                "image_quality":0,
                                "ISO":0,
                                "lenses":0,
                                "light":0,
                                "long_exposure":0,
                                "low_light_performance":0,
                                "manual_settings":0,
                                "megapixels":0,
                                "midprice":0,
                                "mirrorless":0,
                                "movement":1,
                                "optical_zoom":0,
                                "pink":0,
                                "point_and_shoot":0,
                                "preset_modes":0,
                                "price":0,
                                "purple":0,
                                "red":0,
                                "sensor_size":0,
                                "shutter_speed":0,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":0,
                                "white":0},
                      "everyday_photos":{"accessories":0,
                                "artistic_effects":1,
                                "autofocus_points":0,
                                "black":0,
                                "bulb_mode":0,
                                "cheap":0,
                                "color":0,
                                "crop_sensor":0,
                                "depth_of_field":0,
                                "DSLR":0,
                                "ease_of_use":1,
                                "effects":1,
                                "expensive":0,
                                "exposure":0,
                                "full_frame":0,
                                "glamour_retouch_effects":1,
                                "high_performance":0,
                                "image_quality":1,
                                "ISO":0,
                                "lenses":0,
                                "light":0,
                                "long_exposure":0,
                                "low_light_performance":0,
                                "manual_settings":0,
                                "megapixels":0,
                                "midprice":0,
                                "mirrorless":0,
                                "movement":0,
                                "optical_zoom":0,
                                "pink":0,
                                "point_and_shoot":1,
                                "preset_modes":1,
                                "price":0,
                                "purple":0,
                                "red":0,
                                "sensor_size":0,
                                "shutter_speed":0,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":0,
                                "white":0},
                      "novice":{"accessories":0,
                                "artistic_effects":1,
                                "autofocus_points":0,
                                "black":0,
                                "bulb_mode":0,
                                "cheap":1,
                                "color":0,
                                "crop_sensor":0,
                                "depth_of_field":0,
                                "DSLR":0,
                                "ease_of_use":1,
                                "effects":1,
                                "expensive":0,
                                "exposure":0,
                                "full_frame":0,
                                "glamour_retouch_effects":1,
                                "high_performance":0,
                                "image_quality":0,
                                "ISO":0,
                                "lenses":0,
                                "light":0,
                                "long_exposure":0,
                                "low_light_performance":0,
                                "manual_settings":0,
                                "megapixels":0,
                                "midprice":0,
                                "mirrorless":0,
                                "movement":0,
                                "optical_zoom":0,
                                "pink":0,
                                "point_and_shoot":1,
                                "preset_modes":1,
                                "price":0,
                                "purple":0,
                                "red":0,
                                "sensor_size":0,
                                "shutter_speed":0,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":0,
                                "white":0},
                      "expert":{"accessories":0,
                                "artistic_effects":0,
                                "autofocus_points":0,
                                "black":0,
                                "bulb_mode":1,
                                "cheap":0,
                                "color":0,
                                "crop_sensor":0,
                                "depth_of_field":0,
                                "DSLR":0,
                                "ease_of_use":0,
                                "effects":0,
                                "expensive":0,
                                "exposure":0,
                                "full_frame":1,
                                "glamour_retouch_effects":0,
                                "high_performance":1,
                                "image_quality":0,
                                "ISO":1,
                                "lenses":1,
                                "light":0,
                                "long_exposure":1,
                                "low_light_performance":0,
                                "manual_settings":1,
                                "megapixels":0,
                                "midprice":0,
                                "mirrorless":0,
                                "movement":0,
                                "optical_zoom":0,
                                "pink":0,
                                "point_and_shoot":0,
                                "preset_modes":0,
                                "price":0,
                                "purple":0,
                                "red":0,
                                "sensor_size":0,
                                "shutter_speed":0,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":0,
                                "white":0},
                      "art_school":{"accessories":0,
                                "artistic_effects":0,
                                "autofocus_points":0,
                                "black":0,
                                "bulb_mode":0,
                                "cheap":1,
                                "color":0,
                                "crop_sensor":0,
                                "depth_of_field":0,
                                "DSLR":0,
                                "ease_of_use":0,
                                "effects":0,
                                "expensive":0,
                                "exposure":0,
                                "full_frame":1,
                                "glamour_retouch_effects":0,
                                "high_performance":1,
                                "image_quality":0,
                                "ISO":1,
                                "lenses":1,
                                "light":0,
                                "long_exposure":1,
                                "low_light_performance":0,
                                "manual_settings":1,
                                "megapixels":0,
                                "midprice":1,
                                "mirrorless":0,
                                "movement":0,
                                "optical_zoom":0,
                                "pink":0,
                                "point_and_shoot":0,
                                "preset_modes":1,
                                "price":0,
                                "purple":0,
                                "red":0,
                                "sensor_size":0,
                                "shutter_speed":0,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":0,
                                "white":0},
                      "pets":{"accessories":0,
                                "artistic_effects":1,
                                "autofocus_points":1,
                                "black":0,
                                "bulb_mode":0,
                                "cheap":0,
                                "color":0,
                                "crop_sensor":0,
                                "depth_of_field":0,
                                "DSLR":0,
                                "ease_of_use":0,
                                "effects":1,
                                "expensive":0,
                                "exposure":0,
                                "full_frame":0,
                                "glamour_retouch_effects":1,
                                "high_performance":0,
                                "image_quality":1,
                                "ISO":0,
                                "lenses":0,
                                "light":0,
                                "long_exposure":0,
                                "low_light_performance":0,
                                "manual_settings":0,
                                "megapixels":0,
                                "midprice":0,
                                "mirrorless":0,
                                "movement":1,
                                "optical_zoom":0,
                                "pink":0,
                                "point_and_shoot":0,
                                "preset_modes":1,
                                "price":0,
                                "purple":0,
                                "red":0,
                                "sensor_size":0,
                                "shutter_speed":0,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":0,
                                "white":0}
                      }


customerTypes = featuresForPicType.keys()

features = featureValues["NIKON"].keys()

custQuesTopToShkpRespTopMap = {"accessories":"accessories",
                                "artistic_effects":"effects",
                                "autofocus_points":"autofocus_points",
                                "black":"color",
                                "bulb_mode":"bulb_mode",
                                "cheap":"price",
                                "color":"color",
                                "crop_sensor":"sensor_size",
                                "depth_of_field":"depth_of_field",
                                "DSLR":"DSLR",
                                "ease_of_use":"ease_of_use",
                                "effects":"effects",
                                "expensive":"price",
                                "exposure":"exposure",
                                "full_frame":"sensor_size",
                                "glamour_retouch_effects":"effects",
                                "high_performance":"high_performance",
                                "image_quality":"image_quality",
                                "ISO":"ISO",
                                "lenses":"lenses",
                                "light":"weight",
                                "long_exposure":"exposure",
                                "low_light_performance":"low_light_performance",
                                "manual_settings":"manual_settings",
                                "megapixels":"megapixels", # ?
                                "midprice":"price",
                                "mirrorless":"mirrorless",
                                "movement":"movement",
                                "optical_zoom":"optical_zoom",
                                "pink":"color",
                                "point_and_shoot":"point_and_shoot",
                                "preset_modes":"preset_modes",
                                "price":"price",
                                "purple":"color",
                                "red":"color",
                                "sensor_size":"sensor_size",
                                "shutter_speed":"exposure",
                                "touch_screen":"touch_screen",
                                "video":"video",
                                "warranty":"warranty",
                                "weight":"weight",
                                "white":"color",
                                "exposure_what":"exposure",
                                "glamour_retouch_effects_what":"effects",
                                "megapixels_what":"megapixels",
                                "preset_modes_what":"preset_modes"}

shkpHasInfo = {"NIKON":{"accessories":0,
                                "artistic_effects":0,
                                "autofocus_points":0,
                                "black":1,
                                "bulb_mode":0,
                                "cheap":1,
                                "color":1,
                                "crop_sensor":0,
                                "depth_of_field":0,
                                "DSLR":0,
                                "ease_of_use":1,
                                "effects":1,
                                "expensive":1,
                                "exposure":0,
                                "full_frame":1,
                                "glamour_retouch_effects":1,
                                "high_performance":0,
                                "image_quality":1,
                                "ISO":0,
                                "lenses":0,
                                "light":1,
                                "long_exposure":0,
                                "low_light_performance":1,
                                "manual_settings":1,
                                "megapixels":1,
                                "midprice":1,
                                "mirrorless":0,
                                "movement":0,
                                "optical_zoom":1,
                                "pink":1,
                                "point_and_shoot":1,
                                "preset_modes":1,
                                "price":1,
                                "purple":1,
                                "red":1,
                                "sensor_size":1,
                                "shutter_speed":1,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":1,
                                "white":1,
                                "exposure_what":0,
                                "glamour_retouch_effects_what":1,
                                "megapixels_what":1,
                                "preset_modes_what":1},
                "SONY":{"accessories":0,
                                "artistic_effects":1,
                                "autofocus_points":1,
                                "black":1,
                                "bulb_mode":1,
                                "cheap":1,
                                "color":1,
                                "crop_sensor":1,
                                "depth_of_field":0,
                                "DSLR":1,
                                "ease_of_use":1,
                                "effects":1,
                                "expensive":1,
                                "exposure":1,
                                "full_frame":1,
                                "glamour_retouch_effects":0,
                                "high_performance":1,
                                "image_quality":1,
                                "ISO":1,
                                "lenses":1,
                                "light":1,
                                "long_exposure":1,
                                "low_light_performance":1,
                                "manual_settings":1,
                                "megapixels":0,
                                "midprice":1,
                                "mirrorless":1,
                                "movement":1,
                                "optical_zoom":1,
                                "pink":1,
                                "point_and_shoot":1,
                                "preset_modes":1,
                                "price":1,
                                "purple":1,
                                "red":1,
                                "sensor_size":1,
                                "shutter_speed":1,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":1,
                                "white":1,
                                "exposure_what":1,
                                "glamour_retouch_effects_what":1,
                                "megapixels_what":1,
                                "preset_modes_what":1},
                "CANON":{"accessories":0,
                                "artistic_effects":0,
                                "autofocus_points":1,
                                "black":1,
                                "bulb_mode":1,
                                "cheap":1,
                                "color":1,
                                "crop_sensor":1,
                                "depth_of_field":0,
                                "DSLR":1,
                                "ease_of_use":1,
                                "effects":0,
                                "expensive":1,
                                "exposure":1,
                                "full_frame":1,
                                "glamour_retouch_effects":0,
                                "high_performance":1,
                                "image_quality":1,
                                "ISO":1,
                                "lenses":1,
                                "light":1,
                                "long_exposure":1,
                                "low_light_performance":1,
                                "manual_settings":1,
                                "megapixels":0,
                                "midprice":1,
                                "mirrorless":0,
                                "movement":1,
                                "optical_zoom":1,
                                "pink":1,
                                "point_and_shoot":0,
                                "preset_modes":1,
                                "price":1,
                                "purple":1,
                                "red":1,
                                "sensor_size":1,
                                "shutter_speed":1,
                                "touch_screen":0,
                                "video":0,
                                "warranty":0,
                                "weight":1,
                                "white":1,
                                "exposure_what":1,
                                "glamour_retouch_effects_what":1,
                                "megapixels_what":1,
                                "preset_modes_what":1}
                }

additionalQuestionTopics = ["exposure_what",
                            "glamour_retouch_effects_what",
                            "megapixels_what",
                            "preset_modes_what"]


customerActions = ["C_ENTERS",
                   "C_GREETS",
                   "C_WALKS_TO_CAMERA",
                   "C_EXAMINES_CAMERA", 
                   "C_SILENT_OR_BACKCHANNEL", 
                   "C_THANK_YOU",
                   "C_THINK_IT_OVER",
                   "C_DECIDE_TO_BUY", 
                   "C_LEAVES",
                   "C_LOOKING_FOR_A_CAMERA",
                   "C_LOOKING_FOR_A_CAMERA_WITH_X",
                   "C_JUST_BROWSING",
                   "C_QUESTION_ABOUT_FEATURE",
                   "C_ASKS_FOR_SOMETHING_WITH_MORE_X",
                   "C_ANSWERS_QUESTION_X"]

shopkeeperActions = ["S_NONE",
                     "S_GREETS",
                     "S_THANK_YOU",
                     "S_RETURNS_TO_COUNTER",
                     "S_ASKS_QUESTION_X",
                     "S_LET_ME_KNOW_IF_YOU_NEED_HELP",
                     "S_ANYTHING_ELSE_I_CAN_HELP_WITH",
                     #"S_EXPLAIN_STORE_LAYOUT",
                     "S_GREET_RESPONSE",
                     "S_NOT_SURE",
                     "S_TRY_IT_OUT",
                     "S_YOURE_WELCOME",
                     "S_ANSWERS_QUESTION_ABOUT_FEATURE",
                     "S_INTRODUCES_FEATURE",
                     "S_INTRODUCES_CAMERA",
                     "S_THIS_IS_THE_MOST_X"]


# list the valid topics for each customer action type

# C_LOOKING_FOR_A_CAMERA_WITH_X
features_lookingForACameraWithX = ["autofocus_points",
                                "black",
                                "bulb_mode",
                                "cheap",
                                "color",
                                "DSLR",
                                "ease_of_use",
                                "effects",
                                "expensive",
                                "full_frame",
                                "high_performance",
                                "image_quality",
                                "ISO",
                                "lenses",
                                "light",
                                "long_exposure",
                                "low_light_performance",
                                "manual_settings",
                                "megapixels",
                                "midprice",
                                "mirrorless",
                                "movement",
                                "optical_zoom",
                                "pink",
                                "point_and_shoot",
                                "preset_modes",
                                "purple",
                                "red",
                                "white"]

# C_QUESTION_ABOUT_FEATURE
features_questionAboutFeature = ["artistic_effects",
                                "autofocus_points",
                                "black",
                                "bulb_mode",
                                "color",
                                "DSLR",
                                "ease_of_use",
                                "effects",
                                "exposure",
                                #"full_frame",
                                "glamour_retouch_effects",
                                "high_performance",
                                "image_quality",
                                "ISO",
                                "lenses",
                                "long_exposure",
                                "low_light_performance",
                                "manual_settings",
                                "megapixels",
                                "mirrorless",
                                "movement",
                                "optical_zoom",
                                "pink",
                                "point_and_shoot",
                                "preset_modes",
                                "price",
                                "purple",
                                "red",
                                "sensor_size",
                                "weight",
                                "white"]

# S_ANSWERS_QUESTION_ABOUT_FEATURE
features_answersQuestionAboutFeature = features_questionAboutFeature


# C_ASKS_FOR_SOMETHING_WITH_MORE_X
features_asksForSomethingWithMoreX = ["autofocus_points",
                                        "cheap",
                                        "color",
                                        "ease_of_use",
                                        "effects",
                                        "expensive",
                                        "high_performance",
                                        "image_quality",
                                        "ISO",
                                        "light",
                                        "long_exposure",
                                        "low_light_performance",
                                        "manual_settings",
                                        "megapixels",
                                        "movement",
                                        "optical_zoom",
                                        "preset_modes"]

# C_ANSWERS_QUESTION_X
features_answersQuestionX = features_lookingForACameraWithX


# S_INTRODUCES_FEATURES
features_introducesFeatures = ["artistic_effects",
                                "autofocus_points",
                                "bulb_mode",
                                "color",
                                "DSLR",
                                "ease_of_use",
                                "long_exposure",
                                "glamour_retouch_effects",
                                "high_performance",
                                "image_quality",
                                "ISO",
                                "low_light_performance",
                                "manual_settings",
                                "megapixels",
                                "mirrorless",
                                "movement",
                                "optical_zoom",
                                "point_and_shoot",
                                "preset_modes",
                                "price",
                                "sensor_size",
                                "weight"]



# possible APPEND mem-dep feature topics
features_forMemdepTopics = ["artistic_effects",
                                "autofocus_points",
                                "black",
                                "bulb_mode",
                                "cheap",
                                "color",
                                "crop_sensor",
                                "depth_of_field",
                                "DSLR",
                                "ease_of_use",
                                "effects",
                                "expensive",
                                "exposure",
                                "full_frame",
                                "glamour_retouch_effects",
                                "high_performance",
                                "image_quality",
                                "ISO",
                                "lenses",
                                "light",
                                "long_exposure",
                                "low_light_performance",
                                "manual_settings",
                                "megapixels",
                                "midprice",
                                "mirrorless",
                                "movement",
                                "optical_zoom",
                                "pink",
                                "point_and_shoot",
                                "preset_modes",
                                "price",
                                "purple",
                                "red",
                                "sensor_size",
                                "shutter_speed",
                                "touch_screen",
                                "video",
                                "warranty",
                                "weight",
                                "white"]


IS_MEM_DEP_PROB = 0.50
S_ASKS_QUESTION_X_PROB = 0.5



fieldnames = ["CUSTOMER_IS_BROWSING",
              "CURRENT_CAMERA_OF_CONVERSATION",
              "MEMORY_SEQUENCE",
              "PREVIOUS_CAMERAS_OF_CONVERSATION",
              "PREVIOUS_FEATURES_OF_CONVERSATION",
              "TRIAL",
              "TURN_COUNT",
              "CUSTOMER_REQUEST_COUNT",
              
              "CUSTOMER_ACTION",
              "OUTPUT_SHOPKEEPER_ACTION",
              
              "SHOPKEEPER_MEM_DEP",
              "MEM_DEP_TOPIC",
              "SHOPKEEPER_TOPIC",
              "CUSTOMER_TOPIC",
              
              "CUSTOMER_SPEECH",
              "CUSTOMER_KEYWORDS",
              "SHOPKEEPER_SPEECH",
              "SHOPKEEPER_KEYWORDS",
              
              "SPATIAL_STATE",
              "STATE_TARGET",
              
              "OUTPUT_SPATIAL_STATE",
              "OUTPUT_STATE_TARGET",
              
              "CUSTOMER_TYPE",
              "CUSTOMER_FROM_MOTION",
              "CUSTOMER_TO_MOTION",
              "CUSTOMER_LOCATION",
              "SHOPKEEPER_FROM_MOTION",
              "SHOPKEEPER_TO_MOTION",
              "SHOPKEEPER_LOCATION",
              "CUSTOMER_KEYWORDS",
              
              "OUTPUT_CUSTOMER_FROM_MOTION",
              "OUTPUT_CUSTOMER_TO_MOTION",
              "OUTPUT_CUSTOMER_LOCATION",
              "OUTPUT_SHOPKEEPER_FROM_MOTION",
              "OUTPUT_SHOPKEEPER_TO_MOTION",
              "OUTPUT_SHOPKEEPER_LOCATION"
              ]


def utterance_to_keyword_map():
    
    #
    # read in customer to keyword map since the keywords aren't contained in the cluster def files
    #
    uttToKeywordMap = {}
    
    with open(tools.dataDir + "/combined_from_curiosity/ClusterTrainer_ClusterSequence_detailed.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            
            if row["CUSTOMER_KEYWORDS"] != "NO_KEYWORDS":
                uttToKeywordMap[row["CUSTOMER_SPEECH"]] = row["CUSTOMER_KEYWORDS"].split(";")
            else:
                uttToKeywordMap[row["CUSTOMER_SPEECH"]] = []
            
            if row["SHOPKEEPER_KEYWORDS"] != "NO_KEYWORDS":
                uttToKeywordMap[row["SHOPKEEPER_SPEECH"]] = row["SHOPKEEPER_KEYWORDS"].split(";")
            else:
                uttToKeywordMap[row["SHOPKEEPER_SPEECH"]] = []
    
    uttToKeywordMap[""] = []
    
    return uttToKeywordMap


def read_customer_cluster_file(filename):
    
    filename = tools.modelDir + "/annotated speech clusters - passive proactive camera customer - tri stm - 2 wgt kw - mc2 - stopwords 1 - mcs 5 - noPam.csv"
    
    customerActionMap = {}
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        hits = []
        
        for row in reader:
            hits.append(row)
        
        # the data has to be in this order to make proper use of the annotations
        #hits.sort(key=lambda x: (x["Cluster.ID"], x["Is.Representative"]), reverse=True)
        
        
        action = None
        topic = None
        badCluster = None
        
        for row in hits:
            if row["Utterance.ID"] == "271":
                pass
            
            ##
            if row["Is.Representative"] == "1":
                if row["BAD_CLUSTER"] != "BAD":
                    action = row["ACTION"]
                    topic = row["TOPIC"]
                    badCluster = False
                    
                else:
                    action = None
                    topic = None
                    badCluster = True
            
            ##
            act = None
            top = None
            
            if row["UTTERANCE_DOESNT_BELONG"] == "1" or row["ACTION"] != "":
                act = row["ACTION"]
                top = row["TOPIC"]
            
            elif badCluster and row["ACTION"] != "":
                act = row["ACTION"]
                top = row["TOPIC"]

            elif action != None:
                act = action
                top = topic
            
            
            ##
            if act != None:
                if act not in customerActionMap:
                    customerActionMap[act] = {}
                
                if top not in customerActionMap[act]:
                    customerActionMap[act][top] = []
                            
                customerActionMap[act][top].append(row)
        
    return customerActionMap



def read_shopkeeper_cluster_file(filename):
    
    filename = tools.modelDir + "/annotated speech clusters - passive proactive camera shopkeeper - tri stm - 2 wgt kw - mc2 - stopwords 1 - mcs 4 - noPam.csv"
    
    shopkeeperActionMap = {}
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        hits = []
        
        for row in reader:
            hits.append(row)
        
        # the data has to be in this order to make proper use of the annotations
        hits.sort(key=lambda x: (x["Cluster.ID"], x["Is.Representative"]), reverse=True)
        
        
        action = None
        topic = None
        cameraOfConversation = None
        memDep = None
        memDepTopic = None
        badCluster = None
        
        for row in hits:
            
            if row["Utterance.ID"] == "4772":
                pass
            
            
            ##
            if row["Is.Representative"] == "1":
                if row["BAD_CLUSTER"] != "BAD":
                    action = row["ACTION"]
                    topic = row["TOPIC"]
                    cameraOfConversation = row["CAMERA_OF_CONVERSATION"]
                    memDep = row["MEM_DEP"]
                    memDepTopic = row["MEM_DEP_TOPIC"]
                    badCluster = False
                
                else:
                    action = None
                    topic = None
                    cameraOfConversation = None
                    memDep = None
                    memDepTopic = None
                    badCluster = None
            
            ##
            act = None
            top = None
            coc = None
            md = None
            mdt = None
            
            if row["ACTION"] != "": # row["UTTERANCE_DOESNT_BELONG"] == "1" or 
                act = row["ACTION"]
                top = row["TOPIC"]
                coc = row["CAMERA_OF_CONVERSATION"]
                md = row["MEM_DEP"]
                mdt = row["MEM_DEP_TOPIC"]
            
            #elif badCluster and row["ACTION"] != "":
            #    act = row["ACTION"]
            #    top = row["TOPIC"]
            #    coc = row["CAMERA_OF_CONVERSATION"]
            #    md = row["MEM_DEP"]
            #    mdt = row["MEM_DEP_TOPIC"]

            elif action != None:
                act = action
                top = topic
                coc = cameraOfConversation
                md = memDep
                mdt = memDepTopic
            
            ##
            if act != None:
                
                # some shopkeeper utterances are applicable for more that one camera
                cocs = coc.split(",")
                for coc in cocs:
                
                    if act not in shopkeeperActionMap:
                        shopkeeperActionMap[act] = {}
                    
                    if top not in shopkeeperActionMap[act]:
                        shopkeeperActionMap[act][top] = {}
                    
                    if coc not in shopkeeperActionMap[act][top]:
                        shopkeeperActionMap[act][top][coc] = {}
                    
                    if md not in shopkeeperActionMap[act][top][coc]:
                        shopkeeperActionMap[act][top][coc][md] = {}
                    
                    if mdt not in shopkeeperActionMap[act][top][coc][md]:
                        shopkeeperActionMap[act][top][coc][md][mdt] = []
                    
                    shopkeeperActionMap[act][top][coc][md][mdt].append(row)
    
    return shopkeeperActionMap



def read_customer_utterance_file(filename):
    filename = tools.modelDir + "customer_utterance_data_2018-4-5.csv"
    
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
    

def read_shopkeeper_utterance_file(filename):
    
    shopkeeperUtteranceMap = {}
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            act = row["ACTION"]
            top = row["TOPIC"]
            coc = row["CAMERA_OF_CONVERSATION"]
            md = row["MEM_DEP"]
            mdt = row["MEM_DEP_TOPIC"]
            utt = row["UTTERANCE"]
            
            if act not in shopkeeperUtteranceMap:
                shopkeeperUtteranceMap[act] = {}
            
            if top not in shopkeeperUtteranceMap[act]:
                shopkeeperUtteranceMap[act][top] = {}
            
            if coc not in shopkeeperUtteranceMap[act][top]:
                shopkeeperUtteranceMap[act][top][coc] = {}
            
            if md not in shopkeeperUtteranceMap[act][top][coc]:
                shopkeeperUtteranceMap[act][top][coc][md] = {}
            
            if mdt not in shopkeeperUtteranceMap[act][top][coc][md]:
                shopkeeperUtteranceMap[act][top][coc][md][mdt] = []
            
            
            if utt != "":
                shopkeeperUtteranceMap[act][top][coc][md][mdt].append(utt)
    
    
    for act in shopkeeperUtteranceMap:
        for top in shopkeeperUtteranceMap[act]:
            for coc in shopkeeperUtteranceMap[act][top]:
                for md in shopkeeperUtteranceMap[act][top][coc]:
                    for mdt in shopkeeperUtteranceMap[act][top][coc][md]:
                        if len(shopkeeperUtteranceMap[act][top][coc][md][mdt]) == 0:
                            shopkeeperUtteranceMap[act][top][coc][md][mdt].append("")
    
    return shopkeeperUtteranceMap


def generate_customer_utterance_file():
    
    customerActionMap = read_customer_cluster_file(None)
    minNumUtts = 1
    
    hits = []
    
    for act in customerActionMap:
        for top in customerActionMap[act]:
            for info in customerActionMap[act][top]:
                
                row = {"ACTION":act,
                       "TOPIC":top,
                       "UTTERANCE_ID":info["Utterance.ID"],
                       "UTTERANCE":info["Utterance"]}
                
                hits.append(row)
            
            
            # add minimum number of empty slots for actions not represented in the h-h data
            for i in range(max(0, minNumUtts - len(customerActionMap[act][top]))):
                
                row = {"ACTION":act,
                       "TOPIC":top,
                       "UTTERANCE_ID":"-1",
                       "UTTERANCE":""}
                
                hits.append(row)
        
        # add empty slots for action, topic combinations missing from the h-h data
        missingTopics = [top for top in features+additionalQuestionTopics if top not in customerActionMap[act]]
        
        for top in missingTopics:
            for i in range(minNumUtts):
                
                row = {"ACTION":act,
                       "TOPIC":top,
                       "UTTERANCE_ID":"-1",
                       "UTTERANCE":""}
                
                hits.append(row)
        
        if act == "C_LOOKING_FOR_A_CAMERA_WITH_X" or act == "C_ANSWERS_QUESTION_X":
            missingCustomerTypes = [custType for custType in customerTypes if ("PURPOSE:"+custType) not in customerActionMap[act]]
            
            for custType in missingCustomerTypes:
                for i in range(minNumUtts):
                    
                    row = {"ACTION":act,
                           "TOPIC":"PURPOSE:"+custType,
                           "UTTERANCE_ID":"-1",
                           "UTTERANCE":""}
                    
                    hits.append(row)
            
            
    
    # adding empty slots for missing actions
    missingActions = [act for act in customerActions if act not in customerActionMap]
    
    for act in missingActions:
        for top in features+additionalQuestionTopics:
            row = {"ACTION":act,
                   "TOPIC":top,
                   "UTTERANCE_ID":"-1",
                   "UTTERANCE":""}
            
            hits.append(row)
    
    
    # save
    with open(sessionDir+"/customer_utterance_data.csv", "wb") as csvfile:
        fd = ["ACTION", "TOPIC", "UTTERANCE_ID", "UTTERANCE"]
        
        writer = csv.DictWriter(csvfile, fieldnames=fd)
        writer.writeheader()
        writer.writerows(hits)



def generate_shopkeeper_utterance_file():
    
    # read the annotated shopkeeper utterances
    shopkeeperActionMap = read_shopkeeper_cluster_file(None)
    
    # read the old shopkeeper utterance file
    filename = tools.modelDir + "/shopkeeper_utterance_data_2018-4-10.csv"
    
    shopkeeperActionMap2 = {}
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            if row["ACTION"] not in shopkeeperActionMap2:
                shopkeeperActionMap2[row["ACTION"]] = {}
            if row["TOPIC"] not in shopkeeperActionMap2[row["ACTION"]]:
                shopkeeperActionMap2[row["ACTION"]][row["TOPIC"]] = {}
            if row["CAMERA_OF_CONVERSATION"] not in shopkeeperActionMap2[row["ACTION"]][row["TOPIC"]]:
                shopkeeperActionMap2[row["ACTION"]][row["TOPIC"]][row["CAMERA_OF_CONVERSATION"]] = {}
            if row["MEM_DEP"] not in shopkeeperActionMap2[row["ACTION"]][row["TOPIC"]][row["CAMERA_OF_CONVERSATION"]]:
                shopkeeperActionMap2[row["ACTION"]][row["TOPIC"]][row["CAMERA_OF_CONVERSATION"]][row["MEM_DEP"]] = {}
            if row["MEM_DEP_TOPIC"] not in shopkeeperActionMap2[row["ACTION"]][row["TOPIC"]][row["CAMERA_OF_CONVERSATION"]][row["MEM_DEP"]]:
                shopkeeperActionMap2[row["ACTION"]][row["TOPIC"]][row["CAMERA_OF_CONVERSATION"]][row["MEM_DEP"]][row["MEM_DEP_TOPIC"]] = []
            
            shopkeeperActionMap2[row["ACTION"]][row["TOPIC"]][row["CAMERA_OF_CONVERSATION"]][row["MEM_DEP"]][row["MEM_DEP_TOPIC"]].append(row)
    
    
    # combine the two utterance maps into one
    shopkeeperActionMap3 = {}
    
    for act in shopkeeperActionMap:
        for top in shopkeeperActionMap[act]:
            for coc in shopkeeperActionMap[act][top]:
                for md in shopkeeperActionMap[act][top][coc]:
                    for mdt in shopkeeperActionMap[act][top][coc][md]:
                        for info in shopkeeperActionMap[act][top][coc][md][mdt]:
                            
                            if info["Utterance"] != "":
                                
                                if act not in shopkeeperActionMap3:
                                    shopkeeperActionMap3[act] = {}
                                if top not in shopkeeperActionMap3[act]:
                                    shopkeeperActionMap3[act][top] = {}
                                if coc not in shopkeeperActionMap3[act][top]:
                                    shopkeeperActionMap3[act][top][coc] = {}
                                if md not in shopkeeperActionMap3[act][top][coc]:
                                    shopkeeperActionMap3[act][top][coc][md] = {}
                                if mdt not in shopkeeperActionMap3[act][top][coc][md]:
                                    shopkeeperActionMap3[act][top][coc][md][mdt] = []
                                
                                tempInfo = {"ACTION":act,
                                            "TOPIC":top,
                                            "CAMERA_OF_CONVERSATION":coc,
                                            "MEM_DEP":md,
                                            "MEM_DEP_TOPIC":mdt,
                                            "UTTERANCE_ID":info["Utterance.ID"],
                                            "UTTERANCE":info["Utterance"]}
                            
                            shopkeeperActionMap3[act][top][coc][md][mdt].append(tempInfo)
    
    for act in shopkeeperActionMap2:
        for top in shopkeeperActionMap2[act]:
            for coc in shopkeeperActionMap2[act][top]:
                for md in shopkeeperActionMap2[act][top][coc]:
                    for mdt in shopkeeperActionMap2[act][top][coc][md]:
                        for info in shopkeeperActionMap2[act][top][coc][md][mdt]:
                            
                            if info["UTTERANCE"] != "":
                                
                                if act not in shopkeeperActionMap3:
                                    shopkeeperActionMap3[act] = {}
                                if top not in shopkeeperActionMap3[act]:
                                    shopkeeperActionMap3[act][top] = {}
                                if coc not in shopkeeperActionMap3[act][top]:
                                    shopkeeperActionMap3[act][top][coc] = {}
                                if md not in shopkeeperActionMap3[act][top][coc]:
                                    shopkeeperActionMap3[act][top][coc][md] = {}
                                if mdt not in shopkeeperActionMap3[act][top][coc][md]:
                                    shopkeeperActionMap3[act][top][coc][md][mdt] = []
                                
                                tempInfo = {"ACTION":act,
                                            "TOPIC":top,
                                            "CAMERA_OF_CONVERSATION":coc,
                                            "MEM_DEP":md,
                                            "MEM_DEP_TOPIC":mdt,
                                            "UTTERANCE_ID":info["UTTERANCE_ID"],
                                            "UTTERANCE":info["UTTERANCE"]}
                                
                                shopkeeperActionMap3[act][top][coc][md][mdt].append(tempInfo)
                                
                                
    # make sure there is an empty slot for every unique combination that should have an utterance
    actionsToAdd = []
    
    #
    # actions where the CoC, etc. don't matter
    #
    for act in ["S_NONE", "S_GREETS", "S_THANK_YOU", "S_RETURNS_TO_COUNTER", "S_ASKS_QUESTION_X", "S_LET_ME_KNOW_IF_YOU_NEED_HELP", "S_ANYTHING_ELSE_I_CAN_HELP_WITH", "S_GREET_RESPONSE", "S_NOT_SURE", "S_TRY_IT_OUT", "S_YOURE_WELCOME"]:
        
        if act not in shopkeeperActionMap3:
            tempInfo = {"ACTION":act,
                        "TOPIC":"",
                        "CAMERA_OF_CONVERSATION":"",
                        "MEM_DEP":"",
                        "MEM_DEP_TOPIC":"",
                        "UTTERANCE_ID":"-1",
                        "UTTERANCE":""}
            actionsToAdd.append(tempInfo)
    
    
    #
    # S_THIS_IS_THE_MOST_X
    #
    for coc in cameras:
        for top in featureValues[coc]:
            if featureValues[coc][top] == 1:
                top2 = custQuesTopToShkpRespTopMap[top]
                
                tempInfo = {"ACTION":"S_THIS_IS_THE_MOST_X",
                            "TOPIC":top2,
                            "CAMERA_OF_CONVERSATION":coc,
                            "MEM_DEP":"",
                            "MEM_DEP_TOPIC":"",
                            "UTTERANCE_ID":"-1",
                            "UTTERANCE":""}
                
                if tempInfo not in actionsToAdd:
                    actionsToAdd.append(tempInfo)
    
    #
    # S_INTRODUCES_CAMERA
    #
    for coc in cameras:
        
        # without mem dep
        tempInfo = {"ACTION":"S_INTRODUCES_CAMERA",
                    "TOPIC":"",
                    "CAMERA_OF_CONVERSATION":coc,
                    "MEM_DEP":"",
                    "MEM_DEP_TOPIC":"",
                    "UTTERANCE_ID":"-1",
                    "UTTERANCE":""}
        
        actionsToAdd.append(tempInfo)
        
        # with mem dep
        for custType in featuresForPicType:    
                tempInfo = {"ACTION":"S_INTRODUCES_CAMERA",
                            "TOPIC":"",
                            "CAMERA_OF_CONVERSATION":coc,
                            "MEM_DEP":"APPEND",
                            "MEM_DEP_TOPIC":"PURPOSE:"+custType,
                            "UTTERANCE_ID":"-1",
                            "UTTERANCE":""}
                
                actionsToAdd.append(tempInfo)
    
    
    #
    # S_ANSWERS_QUESTION_ABOUT_FEATURE
    #
    for coc in cameras:
        for top in features+additionalQuestionTopics:
            
            if top in shkpHasInfo[coc] and shkpHasInfo[coc][top] == 1:
                
                top2 = custQuesTopToShkpRespTopMap[top]
                
                # without mem dep
                tempInfo = {"ACTION":"S_ANSWERS_QUESTION_ABOUT_FEATURE",
                            "TOPIC":top2,
                            "CAMERA_OF_CONVERSATION":coc,
                            "MEM_DEP":"",
                            "MEM_DEP_TOPIC":"",
                            "UTTERANCE_ID":"-1",
                            "UTTERANCE":""}
                
                if tempInfo not in actionsToAdd:
                    actionsToAdd.append(tempInfo)
                
                # with mem dep
                for custType in featuresForPicType:
                    if (top in featuresForPicType[custType] and featuresForPicType[custType][top] == 1) or (top2 in featuresForPicType[custType] and featuresForPicType[custType][top2] == 1):
                        if (top in featureValues[coc] and featureValues[coc][top] == 1) or (top2 in featureValues[coc] and featureValues[coc][top2] == 1):
                            
                            tempInfo = {"ACTION":"S_ANSWERS_QUESTION_ABOUT_FEATURE",
                                        "TOPIC":top2,
                                        "CAMERA_OF_CONVERSATION":coc,
                                        "MEM_DEP":"APPEND",
                                        "MEM_DEP_TOPIC":"PURPOSE:"+custType,
                                        "UTTERANCE_ID":"-1",
                                        "UTTERANCE":""}
                            
                            if tempInfo not in actionsToAdd:
                                actionsToAdd.append(tempInfo)
    
    #
    # S_INTRODUCES_FEATURE
    #
    for coc in cameras:
        for top in features:
            
            if top in featureValues[coc] and featureValues[coc][top] == 1:
                
                top2 = custQuesTopToShkpRespTopMap[top]
                
                # without mem dep
                tempInfo = {"ACTION":"S_INTRODUCES_FEATURE",
                            "TOPIC":top2,
                            "CAMERA_OF_CONVERSATION":coc,
                            "MEM_DEP":"",
                            "MEM_DEP_TOPIC":"",
                            "UTTERANCE_ID":"-1",
                            "UTTERANCE":""}
                
                if tempInfo not in actionsToAdd:
                    actionsToAdd.append(tempInfo)
                
                # with mem dep
                for custType in featuresForPicType:
                    if (top in featuresForPicType[custType] and featuresForPicType[custType][top] == 1) or (top2 in featuresForPicType[custType] and featuresForPicType[custType][top2] == 1):
                        
                        tempInfo = {"ACTION":"S_INTRODUCES_FEATURE",
                                    "TOPIC":top2,
                                    "CAMERA_OF_CONVERSATION":coc,
                                    "MEM_DEP":"APPEND",
                                    "MEM_DEP_TOPIC":"PURPOSE:"+custType,
                                    "UTTERANCE_ID":"-1",
                                    "UTTERANCE":""}
                        
                        if tempInfo not in actionsToAdd:
                            actionsToAdd.append(tempInfo)

    
    #
    # add the empty slots to the action map
    #
    for actToAdd in actionsToAdd:        
        if actToAdd["ACTION"] not in shopkeeperActionMap3:
            shopkeeperActionMap3[actToAdd["ACTION"]] = {}
        if actToAdd["TOPIC"] not in shopkeeperActionMap3[actToAdd["ACTION"]]:
            shopkeeperActionMap3[actToAdd["ACTION"]][actToAdd["TOPIC"]] = {}
        if actToAdd["CAMERA_OF_CONVERSATION"] not in shopkeeperActionMap3[actToAdd["ACTION"]][actToAdd["TOPIC"]]:
            shopkeeperActionMap3[actToAdd["ACTION"]][actToAdd["TOPIC"]][actToAdd["CAMERA_OF_CONVERSATION"]] = {}
        if actToAdd["MEM_DEP"] not in shopkeeperActionMap3[actToAdd["ACTION"]][actToAdd["TOPIC"]][actToAdd["CAMERA_OF_CONVERSATION"]]:
            shopkeeperActionMap3[actToAdd["ACTION"]][actToAdd["TOPIC"]][actToAdd["CAMERA_OF_CONVERSATION"]][actToAdd["MEM_DEP"]] = {}
        if actToAdd["MEM_DEP_TOPIC"] not in shopkeeperActionMap3[actToAdd["ACTION"]][actToAdd["TOPIC"]][actToAdd["CAMERA_OF_CONVERSATION"]][actToAdd["MEM_DEP"]]:
            shopkeeperActionMap3[actToAdd["ACTION"]][actToAdd["TOPIC"]][actToAdd["CAMERA_OF_CONVERSATION"]][actToAdd["MEM_DEP"]][actToAdd["MEM_DEP_TOPIC"]] = []
        
        if len(shopkeeperActionMap3[actToAdd["ACTION"]][actToAdd["TOPIC"]][actToAdd["CAMERA_OF_CONVERSATION"]][actToAdd["MEM_DEP"]][actToAdd["MEM_DEP_TOPIC"]]) == 0:
            shopkeeperActionMap3[actToAdd["ACTION"]][actToAdd["TOPIC"]][actToAdd["CAMERA_OF_CONVERSATION"]][actToAdd["MEM_DEP"]][actToAdd["MEM_DEP_TOPIC"]].append(actToAdd)
    
    
    #
    # save to file
    #
    with open(sessionDir+"/shopkeeper_utterance_data.csv", "wb") as csvfile:
        fd = ["ACTION", "TOPIC", "CAMERA_OF_CONVERSATION", "MEM_DEP", "MEM_DEP_TOPIC", "UTTERANCE_ID", "UTTERANCE"]
        writer = csv.DictWriter(csvfile, fieldnames=fd)
        writer.writeheader()
        
        for act in shopkeeperActionMap3:
            for top in shopkeeperActionMap3[act]:
                for coc in shopkeeperActionMap3[act][top]:
                    for md in shopkeeperActionMap3[act][top][coc]:
                        for mdt in shopkeeperActionMap3[act][top][coc][md]:
                            for info in shopkeeperActionMap3[act][top][coc][md][mdt]:
                                writer.writerow(info)



custActToUttMap = read_customer_utterance_file(None)
#shkpActToUttMap = read_shopkeeper_utterance_file(tools.modelDir + "/shopkeeper_utterance_data_2018-4-10.csv") # old
shkpActToUttMap = read_shopkeeper_utterance_file(tools.modelDir + "shopkeeper_utterance_data_2018-4-11.csv") # new
uttToKeywordMap = utterance_to_keyword_map()







def get_camera_utilities(interactionState, useOnlyLastRequest=False):
    utilities = {"NIKON":0,
                 "SONY":0,
                 "CANON":0}
    
    if len(interactionState.memorySequence) > 0:

        if useOnlyLastRequest:
            pass
            """
            lastRequest = interactionState.memorySequence[-1]
            
            # increase utility for any feature the customer explicitly mentioned
            if x in features:
                for cam in utilities:
                    utilities[cam] += featureValues[cam][x]
            
            # increase utility for any feature related to the desired picture type
            elif x in customerTypes:
                for cam in utilities:
                    for feat, val in featuresForPicType[x].items():                    
                        utilities[cam] += val * featureValues[cam][feat]
            """
        else:
            for x in interactionState.memorySequence:
                
                # increase utility for any feature the customer explicitly mentioned
                if x in features:
                    for cam in utilities:
                        utilities[cam] += featureValues[cam][x]
                
                # increase utility for any feature related to the desired picture type
                elif x in customerTypes:
                    for cam in utilities:
                        for feat, val in featuresForPicType[x].items():                    
                            utilities[cam] += val * featureValues[cam][feat]
        
    return utilities



class InteractionState(object):
    
    def __init__(self, trialId=-1):
        
        self.end = False
        
        self.trialId = str(trialId)
        self.turnCount = 0
        
        self.customerIsBrowsing = False
        self.customerType = None
        
        self.currentCameraOfConversation = None
        self.previousCamerasOfConversation = []
        self.previousFeaturesOfConversation = {"SONY":[], "CANON":[], "NIKON":[]}
        
        self.memorySequence = [] # these contains things relevant to mem-dep actions: desired features, desired picture types
        self.customerRequestCount = 0
        self.sAsksQuestionXProb = S_ASKS_QUESTION_X_PROB
        
        self.customerAction = None
        self.customerTopic = None
        
        self.customerSpeech = None
        self.customerKeywords = None
        
        self.spatialState = None
        self.stateTarget = None
        self.customerFromMotion = None
        self.customerToMotion = None
        self.customerLocation = None
        
        
        self.outputShopkeeperAction = None
        self.shopkeeperMemDep = None
        self.memDepTopic = None
        self.shopkeeperTopic = None
        
        self.shopkeeperSpeech = None
        self.shopkeeperKeywords = None
        
        self.outputSpatialState = None
        self.outputStateTarget = None
        self.outputShopkeeperFromMotion = None
        self.outputShopkeeperToMotion = None
        self.outputShopkeeperLocation = None
    
    def update_on_customer_action(self, prevIntState, custrActMap):
        pass     
    
    def update_on_shopkeeper_action(self, prevIntState, shkpActMap):        
        pass
    
        
    def to_dict(self):
        
        dictionary = {"CUSTOMER_IS_BROWSING":self.customerIsBrowsing,
                      "CURRENT_CAMERA_OF_CONVERSATION":self.currentCameraOfConversation,
                       "MEMORY_SEQUENCE":self.memorySequence, 
                       "PREVIOUS_CAMERAS_OF_CONVERSATION":self.previousCamerasOfConversation,
                       "PREVIOUS_FEATURES_OF_CONVERSATION":self.previousFeaturesOfConversation,
                       "TRIAL":self.trialId,
                       "TURN_COUNT":self.turnCount,
                       "CUSTOMER_REQUEST_COUNT":self.customerRequestCount,
                       
                       "CUSTOMER_ACTION":self.customerAction,
                       "OUTPUT_SHOPKEEPER_ACTION":self.outputShopkeeperAction,
                       
                       "SHOPKEEPER_MEM_DEP":self.shopkeeperMemDep,
                       "MEM_DEP_TOPIC":self.memDepTopic,
                       "SHOPKEEPER_TOPIC":self.shopkeeperTopic,
                       "CUSTOMER_TOPIC":self.customerTopic,
                       
                       "CUSTOMER_SPEECH":self.customerSpeech,
                       "CUSTOMER_KEYWORDS":self.customerKeywords,
                       "SHOPKEEPER_SPEECH":self.shopkeeperSpeech,
                       "SHOPKEEPER_KEYWORDS":self.shopkeeperKeywords,
                       
                       "SPATIAL_STATE":self.spatialState,
                       "STATE_TARGET":self.stateTarget,
                       
                       "OUTPUT_SPATIAL_STATE":self.outputSpatialState,
                       "OUTPUT_STATE_TARGET":self.outputStateTarget,
                       
                       "CUSTOMER_TYPE":self.customerType,
                       "CUSTOMER_FROM_MOTION":self.customerFromMotion,
                       "CUSTOMER_TO_MOTION":self.customerToMotion,
                       "CUSTOMER_LOCATION":self.customerLocation,
                       "SHOPKEEPER_FROM_MOTION":"",
                       "SHOPKEEPER_TO_MOTION":"",
                       "SHOPKEEPER_LOCATION":"",
                       
                       "OUTPUT_CUSTOMER_FROM_MOTION":"",
                       "OUTPUT_CUSTOMER_TO_MOTION":"",
                       "OUTPUT_CUSTOMER_LOCATION":"",
                       "OUTPUT_SHOPKEEPER_FROM_MOTION":self.outputShopkeeperFromMotion,
                       "OUTPUT_SHOPKEEPER_TO_MOTION":self.outputShopkeeperToMotion,
                       "OUTPUT_SHOPKEEPER_LOCATION":self.outputShopkeeperLocation
                    }
        
        return dictionary
        
        

class CustomerAgent(object):
    
    def __init__(self):
        
        # randomly choose a customer time
        self.type = np.random.choice(customerTypes)
        self.browsing = False
        
        # randomly choose some desired fetures
        
        pass
    
    
    #
    # choose the next customer action and execute
    #
    def perform_action(self, prevIntState):
        
        currIntState = InteractionState(prevIntState.trialId)
        
        # copy the relevant information from the previous interaction state
        currIntState.customerIsBrowsing = prevIntState.customerIsBrowsing
        currIntState.currentCameraOfConversation = prevIntState.currentCameraOfConversation
        currIntState.memorySequence = copy.deepcopy(prevIntState.memorySequence)
        currIntState.sAsksQuestionXProb = prevIntState.sAsksQuestionXProb
        currIntState.previousCamerasOfConversation = copy.deepcopy(prevIntState.previousCamerasOfConversation)
        currIntState.previousFeaturesOfConversation = copy.deepcopy(prevIntState.previousFeaturesOfConversation)
        currIntState.customerRequestCount = prevIntState.customerRequestCount
        currIntState.trialId = prevIntState.trialId
            
        currIntState.turnCount = prevIntState.turnCount + 1
        
        
        if prevIntState.turnCount == 0:
            action = self.action_enters
        
        else:    
        
            if prevIntState.customerAction == "C_ENTERS":
                if self.browsing == True:
                    action = self.action_walks_to_camera
                else:
                    action = np.random.choice([self.action_greets, self.action_walks_to_camera], p=[0.5, 0.5])
            
            elif prevIntState.outputShopkeeperAction == "S_GREETS":
                action = np.random.choice([self.action_looking_for_a_camera, self.action_looking_for_a_camera_with_x, self.action_just_browsing], p=[0.33, 0.33, 0.34])
                
            elif prevIntState.outputShopkeeperAction == "S_ASKS_QUESTION_X":
                action = self.action_answers_question_x
                
            elif prevIntState.outputShopkeeperAction == "S_INTRODUCES_CAMERA":
                action = np.random.choice([self.action_question_about_feature, self.action_silent_or_backchannel], p=[0.66, 0.34])
                
            elif prevIntState.outputShopkeeperAction == "S_LET_ME_KNOW_IF_YOU_NEED_HELP":
                action = self.action_walks_to_camera
            
            elif prevIntState.outputShopkeeperAction == "S_ANSWERS_QUESTION_ABOUT_FEATURE" or prevIntState.outputShopkeeperAction == "S_NOT_SURE":
                if currIntState.customerRequestCount >= 2:
                    action = np.random.choice([self.action_question_about_feature, self.action_silent_or_backchannel, self.action_thank_you], p=[0.55, 0.30, 0.15])
                else:
                    action = np.random.choice([self.action_question_about_feature, self.action_asks_for_something_with_more_x, self.action_silent_or_backchannel, self.action_thank_you], p=[0.5, 0.15, 0.25, 0.10])
            
            elif prevIntState.outputShopkeeperAction == "S_INTRODUCES_FEATURE":
                action = np.random.choice([self.action_silent_or_backchannel, self.action_question_about_feature, self.action_thank_you], p=[0.33, 0.34, 0.33])
        
            elif prevIntState.outputShopkeeperAction == "S_THANK_YOU":
                action = self.action_leaves
                                
            elif prevIntState.outputShopkeeperAction == "S_THIS_IS_THE_MOST_X":
                action = np.random.choice([self.action_question_about_feature, self.action_silent_or_backchannel, self.action_thank_you], p=[0.33, 0.33, 0.34])
                
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
        
        currIntState.customerSpeech = np.random.choice(custActToUttMap[currIntState.customerAction][top])
        
        if currIntState.customerSpeech in uttToKeywordMap:
            currIntState.customerKeywords = ";".join(uttToKeywordMap[currIntState.customerSpeech])
        else:
            currIntState.customerKeywords = "NO_KEYWORD"
        
        if DEBUG:
            print currIntState.customerAction, top
        
        
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
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = "DOOR"
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
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
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = "MIDDLE"
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
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
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = cam
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
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
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        return currIntState
    
    
    
    #
    # C_SILENT_OR_BACKCHANNEL
    #
    def action_silent_or_backchannel(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_SILENT_OR_BACKCHANNEL"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
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
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
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
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        return currIntState
    
    
    #
    # C_DECIDE_TO_BUY
    #
    def action_decides_to_buy(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_DECIDE_TO_BUY"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
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
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = "DOOR"
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
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
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        return currIntState
    
    
    #
    # C_LOOKING_FOR_A_CAMERA_WITH_X
    #
    def action_looking_for_a_camera_with_x(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_LOOKING_FOR_A_CAMERA_WITH_X"
        
        # this could be a specific feature or a picture type
        
        # find if the desired pic type has already been stated
        statedPicTypes = [mem for mem in currIntState.memorySequence if (mem in customerTypes)]
        
        # if the customer has not already stated their desired picture type, state it
        if len(statedPicTypes) == 0:
            currIntState.memorySequence.append(self.type)
            currIntState.customerTopic = "PURPOSE:{}".format(self.type)
            
            currIntState.sAsksQuestionXProb = 0.0
            
        # if the customer has already stated their desired picture type,
        # choose a feature that is related to the customer's desired picture type
        else:
            candidateFeats = [x for x in features_lookingForACameraWithX if featuresForPicType[self.type][x] == 1]
            desiredFeat = np.random.choice(candidateFeats)
            
            currIntState.memorySequence.append(desiredFeat)
            currIntState.customerTopic = desiredFeat #"DESIRED_FEAT:{}".format(desiredFeat)
            
        
        currIntState.customerSpeech = ""
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        return currIntState
    
    
    #
    # C_JUST_BROWSING
    #
    def action_just_browsing(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_JUST_BROWSING"
        
        currIntState.customerIsBrowsing = True
        currIntState.currentCameraOfConversation = "NONE"
        
        currIntState.customerTopic = "NONE"
        currIntState.customerSpeech = ""
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
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
                print "WARNING: Customer is not currently located at a camera!"
        
        # choose a feature to ask about
        
        # which features have not yet been asked about
        candidateFeats = [x for x in features_questionAboutFeature if x not in currIntState.previousFeaturesOfConversation[currIntState.currentCameraOfConversation]]
        candidateFeats += [x for x in additionalQuestionTopics if x not in currIntState.previousFeaturesOfConversation[currIntState.currentCameraOfConversation]]
        
        
        # which features are related to the customer's desired picture type
        relevantFeats = [x for x in features_questionAboutFeature if featuresForPicType[self.type][x] == 1]
        
        # choose a feature related to the customer's desired picture type that has not already been talked about
        unionFeats = [x for x in candidateFeats if x in relevantFeats]
        
        #if len(unionFeats) > 0:
        #    feat = np.random.choice(unionFeats)
        
        # choose a feature that has not already been talked about
        if len(candidateFeats) > 0:
            feat = np.random.choice(candidateFeats)
        
        # reask about a feature related to the desired picture type
        else:
            feat = np.random.choice(relevantFeats)
        
        
        currIntState.previousFeaturesOfConversation[currIntState.currentCameraOfConversation].append(feat)
        
        currIntState.customerTopic = feat
        currIntState.customerSpeech = ""
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
        currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
        
        return currIntState
    
    
    #
    # C_ASKS_FOR_SOMETHING_WITH_MORE_X
    #
    def action_asks_for_something_with_more_x(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_ASKS_FOR_SOMETHING_WITH_MORE_X"
        
        # which features have been talked about
        candidateFeats = [x for x in features_asksForSomethingWithMoreX if x in currIntState.previousFeaturesOfConversation[currIntState.currentCameraOfConversation]]
        
        # which features has the customer already requested more of 
        # try to avoid repetition
        alreadyRequested = [x for x in features_asksForSomethingWithMoreX if x in currIntState.memorySequence]
        
        candidateFeats2 = [x for x in candidateFeats if x not in alreadyRequested]
        
        if len(candidateFeats2) == 0:
            # the customer should ask more questions about features instead
            # of requesting something more...
            currIntState = self.action_question_about_feature(prevIntState, currIntState)
        
        else:
            
            currIntState.currentCameraOfConversation = prevIntState.currentCameraOfConversation
            
            # which features are related to the customer's desired picture type
            relevantFeats = [x for x in features_asksForSomethingWithMoreX if featuresForPicType[self.type][x] == 1]
            
            # which features have been talked about but haven't been requested more of
            candidateFeats = [x for x in features_asksForSomethingWithMoreX if (x in currIntState.previousFeaturesOfConversation[currIntState.currentCameraOfConversation]) and (x not in currIntState.memorySequence)]
            
            # which features that have been talked about but do not meet the customer's
            # requirements for the current camera
            spokenUnmetRequirements = [x for x in candidateFeats if x in relevantFeats and featuresForPicType[self.type][x] == 1]
            
            if len(spokenUnmetRequirements) > 0:
                desiredFeat = np.random.choice(spokenUnmetRequirements)
            
            # just ask for one of the desired features
            else:
                desiredFeat = np.random.choice(relevantFeats)
            
            currIntState.memorySequence.append(desiredFeat)
            
            currIntState.customerTopic = desiredFeat #"DESIRED_FEAT:{}".format(desiredFeat)
            currIntState.customerSpeech = ""
            currIntState.spatialState = ""
            currIntState.stateTarget = ""
            currIntState.customerFromMotion = ""
            currIntState.customerToMotion = ""
            currIntState.customerLocation = prevIntState.outputCustomerLocation
            currIntState.shopkeeperFromMotion = ""
            currIntState.shopkeeperToMotion = ""
            currIntState.shopkeeperLocation = prevIntState.outputShopkeeperLocation
            
            currIntState.customerRequestCount += 1
            
        
        return currIntState
    
    
    #
    # C_ANSWERS_QUESTION_X
    #
    def action_answers_question_x(self, prevIntState, currIntState):
        
        currIntState.customerAction = "C_ANSWERS_QUESTION_X"
        
        #if prevIntState.shopkeeperTopic == "PICS":
        currIntState.memorySequence.append(self.type)
        currIntState.customerTopic = "PURPOSE:{}".format(self.type)
        
        #elif prevIntState.shopkeeperTopic == "FEATS":
        #    pass
            
            
        #elif prevIntState.shopkeeperTopic == "CAMERA_TYPE":
        #    if customerInfo["CUSTOMER_KNOWLEDGE"] == "NOVICE":
        #        currIntState.memorySequence.append("point and shoot")
        #        currIntState.customerTopic = "DESIRED_FEAT:point and shoot"
        #        
        #    else:
        #        currIntState.memorySequence.append("DSLR")
        #        currIntState.customerTopic = "DESIRED_FEAT:DSLR"
        
        
        currIntState.currentCameraOfConversation = prevIntState.currentCameraOfConversation
        
        currIntState.customerSpeech = ""
        currIntState.spatialState = ""
        currIntState.stateTarget = ""
        currIntState.customerFromMotion = ""
        currIntState.customerToMotion = ""
        currIntState.customerLocation = prevIntState.outputCustomerLocation
        currIntState.shopkeeperFromMotion = ""
        currIntState.shopkeeperToMotion = ""
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
                
            elif currIntState.customerAction == "C_LOOKING_FOR_A_CAMERA":
                action = self.action_asks_question_x
                
            elif currIntState.customerAction == "C_LOOKING_FOR_A_CAMERA_WITH_X":
                action = np.random.choice([self.action_asks_question_x, self.action_introduces_camera], p=[currIntState.sAsksQuestionXProb, (1.0-currIntState.sAsksQuestionXProb)])
            
            elif currIntState.customerAction == "C_ANSWERS_QUESTION_X":
                action = self.action_introduces_camera
            
            elif currIntState.customerAction == "C_EXAMINES_CAMERA":
                if currIntState.customerIsBrowsing:
                    action = self.action_none
                else:
                    action = np.random.choice([self.action_none, self.action_greets], p=[0.25, 0.75])
                
            elif currIntState.customerAction == "C_WALKS_TO_CAMERA":
                if currIntState.customerIsBrowsing:
                    action = self.action_none
                else:
                    action = np.random.choice([self.action_none, self.action_greets], p=[0.25, 0.75])
                
            elif currIntState.customerAction == "C_JUST_BROWSING":
                action = self.action_let_me_know_if_you_need_help
                
            elif prevIntState.outputShopkeeperAction == "S_LET_ME_KNOW_IF_YOU_NEED_HELP":
                action = self.action_return_to_counter
                
            elif currIntState.customerAction == "C_QUESTION_ABOUT_FEATURE":
                action = self.action_answers_question_about_feature
                
            elif currIntState.customerAction == "C_ASKS_FOR_SOMETHING_WITH_MORE_X":
                action = self.action_introduces_camera
                
            elif currIntState.customerAction == "C_SILENT_OR_BACKCHANNEL":
                if prevIntState.customerSpeech == "NONE" and currIntState.customerSpeech == "NONE":
                    action = self.action_anything_else_i_can_help_you_with
                else:
                    action = np.random.choice([self.action_asks_question_x, self.action_introduces_feature], p=[currIntState.sAsksQuestionXProb, (1.0-currIntState.sAsksQuestionXProb)])
                
            elif currIntState.customerAction == "C_THANK_YOU":
                action = self.action_thank_you
                
            elif currIntState.customerAction == "C_THINK_IT_OVER":
                action = self.action_thank_you
                
            elif currIntState.customerAction == "C_DECIDE_TO_BUY":
                action = self.action_thank_you
                
            elif currIntState.customerAction == "C_LEAVES":
                action = self.action_return_to_counter
        
        
        # make sure a browsing customer is not greeted twice
        if currIntState.outputShopkeeperAction == "S_GREETS" and currIntState.customerIsBrowsing:
            action = self.action_none
        
        
        currIntState = action(currIntState)
        
        currIntState = self.add_mem_dep(currIntState)
        
        currIntState = self.update_previous_cameras_of_conversation(currIntState)
        

        #
        # choose the shopkeeper speech and keywords
        #
        if currIntState.shopkeeperTopic == "NONE" or currIntState.shopkeeperTopic == "" or currIntState.shopkeeperTopic == None:
            top = ""
        else:
            top = currIntState.shopkeeperTopic
        
        if currIntState.outputShopkeeperAction in ["S_NONE", "S_GREETS", "S_THANK_YOU", "S_RETURNS_TO_COUNTER", "S_ASKS_QUESTION_X", "S_LET_ME_KNOW_IF_YOU_NEED_HELP", "S_ANYTHING_ELSE_I_CAN_HELP_WITH", "S_GREET_RESPONSE", "S_NOT_SURE", "S_TRY_IT_OUT", "S_YOURE_WELCOME"]:
            coc = ""
        else:
            coc = currIntState.currentCameraOfConversation
        
        if currIntState.shopkeeperMemDep == "S_APPENDED_PHRASE_RELATING_TO_MEMORY":
            md ="APPEND"
            mdt = "PURPOSE:"+currIntState.memDepTopic
        else:
            md = ""
            mdt = ""
        
        if DEBUG:
            print currIntState.outputShopkeeperAction, top, coc, md, mdt
        
        
        #try:
        currIntState.shopkeeperSpeech = np.random.choice(shkpActToUttMap[currIntState.outputShopkeeperAction][top][coc][md][mdt])
        #except Exception, e:
        #    print str(e)
            
            
        if currIntState.shopkeeperSpeech in uttToKeywordMap:
            currIntState.shopkeeperKeywords = ";".join(uttToKeywordMap[currIntState.shopkeeperSpeech])
        else:
            currIntState.shopkeeperKeywords = "NO_KEYWORD"
        
        
        
        return currIntState
    
    
    
    #
    # append memory-dependent phrase to shopkeeper speech
    #
    def add_mem_dep(self, currIntState):
           
        if len(currIntState.memorySequence) > 0:
            if currIntState.outputShopkeeperAction == "S_ANSWERS_QUESTION_ABOUT_FEATURE" or currIntState.outputShopkeeperAction == "S_INTRODUCES_FEATURE":
                
                # if the current camera has the feature being talked about
                if currIntState.shopkeeperTopic in featureValues[currIntState.currentCameraOfConversation] and featureValues[currIntState.currentCameraOfConversation][currIntState.shopkeeperTopic] == 1:
                    # and the feature is related to the requested picture type
                    requestedPicType = [x for x in currIntState.memorySequence if x in featuresForPicType]
                    if len(requestedPicType) > 0: # check if the customer has requested a certain picture type
                        if featuresForPicType[requestedPicType[0]][currIntState.shopkeeperTopic] == 1:
                            if np.random.random() < IS_MEM_DEP_PROB:
                                currIntState.shopkeeperMemDep = "S_APPENDED_PHRASE_RELATING_TO_MEMORY"
                                currIntState.memDepTopic = requestedPicType[0]
                
                # if a previous camera also had the feature being talked about and 
                # it was mentioned...
                #currIntState.shopkeeperMemDep = "S_COMPARES_TO_PREVIOUS_CAMERA"
            
            
            elif currIntState.outputShopkeeperAction == "S_INTRODUCES_CAMERA":
                
                # if the current camera is good for the requested picture type,
                # mention it
                camUtil = get_camera_utilities(currIntState)
                sortedCams = [c for c in sorted(camUtil.keys(), key=lambda u: camUtil[u], reverse=True)]
                
                if sortedCams[0] == currIntState.currentCameraOfConversation:
                            
                    if np.random.random() < IS_MEM_DEP_PROB:
                        
                        
                        # select the picture type from the memory sequence that is consistent with this camera
                        requestedPicType = [x for x in currIntState.memorySequence if x in featuresForPicType]
                        if len(requestedPicType) > 0: # check if the customer has requested a certain picture type
                        
                                currIntState.shopkeeperMemDep = "S_APPENDED_PHRASE_RELATING_TO_MEMORY"
                                currIntState.memDepTopic = requestedPicType[0]
                        
                        # select a feature from the memory sequence that is consistent with this camera
                        #else:
                        #    requestedFeat = [x for x in currIntState.memorySequence if x in features and featureValues[currIntState.currentCameraOfConversation][x] == 1]
                        #    if len(requestedFeat) > 0:
                        #        
                        #        currIntState.shopkeeperMemDep = "S_APPENDED_PHRASE_RELATING_TO_MEMORY"
                        #        currIntState.memDepTopic = np.random.choice(requestedFeat)
        
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
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
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
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_THANK_YOU
    #
    def action_thank_you(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_THANK_YOU"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = currIntState.shopkeeperLocation
        
        return currIntState
    
    
    #
    # S_RETURNS_TO_COUNTER
    #
    def action_return_to_counter(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_RETURNS_TO_COUNTER"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = "SERVICE_COUNTER"
        
        return currIntState
    
    
    #
    # S_ASKS_QUESTION_X
    #
    def action_asks_question_x(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_ASKS_QUESTION_X"
        
        # choose to ask about either desired picture type or what type of camera
        knownPicType = [x for x in currIntState.memorySequence if (x in customerTypes)]
        
        # if the shopkeeper does not already know the customer's desired picture type
        if len(knownPicType) == 0:
            # ask what type of pictures the customer wants to take
            currIntState.shopkeeperTopic = "PICS"
            
            currIntState.sAsksQuestionXProb = 0.0
        
        #else:
        #    # ask what type of camera they are looking for
        #    currIntState.shopkeeperTopic = "FEATS"
        #    
        #    currIntState.sAsksQuestionXProb = 0.0
        
        
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    
    #
    # S_LET_ME_KNOW_IF_YOU_NEED_HELP
    #
    def action_let_me_know_if_you_need_help(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_LET_ME_KNOW_IF_YOU_NEED_HELP"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = "SERVICE_COUNTER"
        
        return currIntState
    
    
    #
    # S_ANYTHING_ELSE_I_CAN_HELP_WITH
    #
    def action_anything_else_i_can_help_you_with(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_ANYTHING_ELSE_I_CAN_HELP_WITH"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_EXPLAIN_STORE_LAYOUT
    #
    def action_explain_store_layout(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_EXPLAIN_STORE_LAYOUT"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_GOODBYE
    #
    def action_goodbye(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_GOODBYE"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = "SERVICE_COUNTER"
        
        return currIntState
    
    
    #
    # S_GREET_RESPONSE
    #
    def action_greet_response(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_GREET_RESPONSE"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_NOT_SURE
    #
    def action_not_sure(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_NOT_SURE"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_TRY_IT_OUT
    #
    def action_try_it_out(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_TRY_IT_OUT"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_YOURE_WELCOME
    #
    def action_youre_welcome(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_YOURE_WELCOME"
        
        currIntState.shopkeeperTopic = "NONE"
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_ANSWERS_QUESTION_ABOUT_FEATURE
    #
    def action_answers_question_about_feature(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_ANSWERS_QUESTION_ABOUT_FEATURE"
        
        top = custQuesTopToShkpRespTopMap[currIntState.customerTopic]
        
        if shkpHasInfo[currIntState.currentCameraOfConversation][top] == 0:
            top = ""
            currIntState.outputShopkeeperAction = "S_NOT_SURE"
        
        currIntState.shopkeeperTopic = top
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        return currIntState
    
    
    #
    # S_INTRODUCES_FEATURE
    #
    def action_introduces_feature(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_INTRODUCES_FEATURE"
        
        currFeat = None
        
        knownPicType = [x for x in currIntState.memorySequence if (x in customerTypes)]
        currCamFeats = [x for x in features if featureValues[currIntState.currentCameraOfConversation][x] == 1]
        
        
        # if the shopkeeper knows the customer's desired pic type
        if len(knownPicType) > 0:
            
            # features related to the picture type
            relevantFeats = [x for x in features if featuresForPicType[knownPicType[0]][x] == 1]
            
            # features that haven't been talked about yet
            unspokenFeats = [x for x in features if x not in currIntState.previousFeaturesOfConversation[currIntState.currentCameraOfConversation]]
            
            
            # if not all relevant features of the current camera have been introduced, introduce one
            candidateFeats = [x for x in relevantFeats if (x in unspokenFeats) and (x in currCamFeats)]
            if len(candidateFeats) > 0:
                currFeat = np.random.choice(candidateFeats)
            
            # if all relevant features of the current camera have already been introduced
            else:
                candidateFeats = [x for x in unspokenFeats if x in currCamFeats]
                
                # if there are unrelated features that have not been talked about yet
                if len(candidateFeats) > 0:
                    currFeat = np.random.choice(candidateFeats)
                
                # else, just repeat a random feature
                else:
                    currFeat = np.random.choice(currCamFeats)
            
        
        # if the shopkeeper does not know the customer's desired pic type
        else:
            # features that the current camera have that have not been talked about yet
            candidateFeats = [x for x in currCamFeats if x not in currIntState.previousFeaturesOfConversation[currIntState.currentCameraOfConversation]]
            
            # introduce a random feature that hasn't been talked about yet
            if len(candidateFeats) > 0:
                currFeat = np.random.choice(candidateFeats)
            
            # if all features have already been introduced, restate a random feature
            else:
                currFeat = np.random.choice(currCamFeats)
        
        
        currIntState.shopkeeperTopic = custQuesTopToShkpRespTopMap[currFeat]
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currIntState.customerLocation
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = currIntState.customerLocation
        
        currIntState.previousFeaturesOfConversation[currIntState.currentCameraOfConversation].append(currIntState.shopkeeperTopic)
        
        return currIntState
        
    
    #
    # S_INTRODUCES_CAMERA
    #
    def action_introduces_camera(self, currIntState):
        
        currIntState.outputShopkeeperAction = "S_INTRODUCES_CAMERA"
        
        # choose a camera
        currCamera = None
        
        # which cameras haven't been talked about yet
        candidateCams = [c for c in cameras if c not in currIntState.previousCamerasOfConversation]
        
        
        # customer asked for camera with more of a specific feature
        #
        if currIntState.customerAction == "C_ASKS_FOR_SOMETHING_WITH_MORE_X":
            
            # which cameras have the requested feature
            requestedFeat = currIntState.memorySequence[-1] # get the requested feature
            camsWithX = [x for x in featureValues if featureValues[x][requestedFeat] == 1]
            
            # if there is something with more X, introduce it
            if currIntState.currentCameraOfConversation not in camsWithX:
                
                camUtil = get_camera_utilities(currIntState)
                sortedCams = [c for c in sorted(camUtil.keys(), key=lambda u: camUtil[u], reverse=True)]
                for c in sortedCams:
                    if c in camsWithX:
                        currCamera = c
                        break
            
            # if the current camera is the most X, state that
            elif len(camsWithX) == 1 and currIntState.currentCameraOfConversation in camsWithX:
                currCamera = currIntState.currentCameraOfConversation
                currIntState.outputShopkeeperAction = "S_THIS_IS_THE_MOST_X"
                currIntState.shopkeeperTopic = requestedFeat
                
            # if another camera also has X
            elif len(camsWithX) > 1 and currIntState.currentCameraOfConversation in camsWithX:
                
                # other camera has not yet been introduced
                for c in camsWithX:
                    if c != currIntState.currentCameraOfConversation and c in candidateCams:
                        currCamera = c
                        
                        currIntState.shopkeeperMemDep = "S_COMPARES_TO_PREVIOUS_CAMERA"
                        currIntState.memDepTopic = currIntState.currentCameraOfConversation # compare the camera that is being introduced to the previously spoken about camera
                        currIntState.shopkeeperTopic = requestedFeat
                        break
                
                # all other cameras have already been introduced but at least one other one has X
                if  currCamera == None:
                    currCamera = currIntState.currentCameraOfConversation
                    currIntState.outputShopkeeperAction = "S_THIS_IS_THE_MOST_X"
                    
                    #currIntState.shopkeeperMemDep = "S_COMPARES_TO_PREVIOUS_CAMERA"
                    currIntState.shopkeeperTopic = requestedFeat
                    
                    #for c in camsWithX:
                    #    if c != currIntState.currentCameraOfConversation:
                    #        currIntState.memDepTopic = c
        
        # customer asked for a camera with a certain feature or good for a certain
        # picture type (opening of interaction)
        #
        elif currIntState.customerAction == "C_LOOKING_FOR_A_CAMERA_WITH_X":
            
            # which cameras have the feature or are good for the picture type previously mentioned?
            requested = currIntState.memorySequence[-1] # get the requested feature
            
            if requested in features:
                # customer requested a camera with a certain feature
                
                # which cameras have the requested feature
                requestedFeat = requested
                camsWithX = [c for c in featureValues if featureValues[c][requestedFeat] == 1]
                
                # this action will be in the beginning phase, so just randomly 
                # choose a camera with the feature and introduce it
                currCamera = np.random.choice(camsWithX)
                
            elif requested in featuresForPicType:
                # customer requested a camera good for taking a certain type of pictures
                camUtil = get_camera_utilities(currIntState)
                sortedCams = [c for c in sorted(camUtil.keys(), key=lambda u: camUtil[u], reverse=True)]
                
                # introduce the camera with the highest utility
                currCamera = sortedCams[0]
            
            else:
                print "WARNING: The requested is not a valid feature or picture type!"
            
        
        # customer answered one of the shopkeeper's questions about desired
        # feature or picture type
        #
        elif currIntState.customerAction == "C_ANSWERS_QUESTION_X":
            
            # what are the utilities of each camera based on the requested 
            # features and picture types up till now
            camUtil = get_camera_utilities(currIntState)
            sortedCams = [c for c in sorted(camUtil.keys(), key=lambda u: camUtil[u], reverse=True)]
            
            # which cameras have the previously requested feature specifically
            requested = currIntState.memorySequence[-1]
            camsWithX = copy.deepcopy(cameras)
            
            if requested in features:
                requestedFeat = requested
                camsWithX = [c for c in featureValues if featureValues[c][requestedFeat] == 1]
            
            # introduce the camera with the specifically requested feature that 
            # has the highest utility
            for c in sortedCams:
                if c in camsWithX:
                    
                    if c == currIntState.currentCameraOfConversation:
                        # don't reintroduce the current camera unless it is the
                        # only one with the requested feature
                        if len(camsWithX) == 1:
                            currCamera = c
                            break
                    else:
                        currCamera = c
                        break
        
        
        currIntState.currentCameraOfConversation = currCamera
        
        
        if currIntState.outputShopkeeperAction == "S_INTRODUCES_CAMERA":
            currIntState.shopkeeperTopic = ""
        
        if currIntState.shopkeeperTopic != "" and currIntState.shopkeeperTopic != "NONE" and currIntState.shopkeeperTopic != None:
            currIntState.shopkeeperTopic = custQuesTopToShkpRespTopMap[currIntState.shopkeeperTopic]
        
            
        currIntState.shopkeeperSpeech = ""
        currIntState.outputSpatialState = ""
        currIntState.outputStateTarget = ""
        currIntState.outputCustomerFromMotion = ""
        currIntState.outputCustomerToMotion = ""
        currIntState.outputCustomerLocation = currCamera
        currIntState.outputShopkeeperFromMotion = ""
        currIntState.outputShopkeeperToMotion = ""
        currIntState.outputShopkeeperLocation = currCamera
        
        if currIntState.outputShopkeeperLocation == "MIDDLE":
            pass
        
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
        
        #print currIntState.turnCount
        
        interaction.append(currIntState.to_dict())
        
        
        if currIntState.outputShopkeeperAction == "S_INTRODUCES_CAMERA" and currIntState.outputShopkeeperLocation == "MIDDLE":
            pass
        
        
        if currIntState.customerAction == "C_LEAVES":
            break
        
        prevIntState = copy.deepcopy(currIntState)
        
    
    return interaction



def simulate_n_interactions(n, flatten=True):
    
    interactions = []
    
    for i in range(n):
        if flatten:
            interactions += simulate_interaction(i, i)
        else:
            interactions.append(simulate_interaction(i, i))
        
        if DEBUG:
            print
    
    return interactions


def analysis(simulatedInteractions):
    # count the proportion of memory-dependent actions, etc.
    numTurnsPerInteraction = []
    numMemDepActions = []
    
    for interaction in simulatedInteractions:
        numTurnsPerInteraction.append(len(interaction))
        numMemDepActions.append(0)
        
        for turn in interaction:
            if turn["SHOPKEEPER_MEM_DEP"] == "S_APPENDED_PHRASE_RELATING_TO_MEMORY":
                numMemDepActions[-1] += 1
    
    print
    print "Num Interactions", len(simulatedInteractions)
    print "Ave num turns per interaction", np.average(numTurnsPerInteraction)
    print "Std num turns per interaction", np.std(numTurnsPerInteraction)
    
    print "Perc. Interactions that contain mem-dep", (len(simulatedInteractions) - numMemDepActions.count(0)) / float(len(simulatedInteractions))
    
    print "Ave num mem-dep actions per interaction (app. phrase)", np.average(numMemDepActions)
    print "Std num mem-dep actions per interaction (app. phrase)", np.std(numMemDepActions) 
    
    print "Total num turns", np.sum(numTurnsPerInteraction)
    print "Total num mem-dep actions (app. phrase)", np.sum(numMemDepActions)
    print "Proportion mem-dep actions (app. phrase)", float(np.sum(numMemDepActions)) / np.sum(numTurnsPerInteraction)
        


if __name__ == "__main__":
    
    
    sessionDir = tools.create_session_dir("advancedSimulator6")
    
    global DEBUG
    DEBUG = False
    
    print "started"
    
    #generate_shopkeeper_utterance_file()
    
    #global shkpActToUttMap
    #shkpActToUttMap = read_shopkeeper_utterance_file(sessionDir + "/shopkeeper_utterance_data.csv")
    
    
    interactions = simulate_n_interactions(100, flatten=False)
    
    #
    # save to file
    #
    
    with open(sessionDir+"/simulated data.csv", "wb") as csvfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for interaction in interactions:
            for row in interaction:
                writer.writerow(row)
    
    
    analysis(interactions)
    
    
    print "Camera utilities per picture type:"
    for picType, featVals in featuresForPicType.items():
        
        intState = InteractionState()
        intState.memorySequence.append(picType)
        
        print picType
        print get_camera_utilities(intState)
    
    
    print "finished"
    
    
    count = 0
    
    for f in features_questionAboutFeature:
        f = custQuesTopToShkpRespTopMap[f]
        
        for p in featuresForPicType:
            
            for c in cameras:
                
                if featuresForPicType[p][f] == 1 and featureValues[c][f] == 1:
                    count += 1
    
    for f in features_introducesFeatures:
        f = custQuesTopToShkpRespTopMap[f]
        
        for p in featuresForPicType:
            
            for c in cameras:
                
                if featuresForPicType[p][f] == 1 and featureValues[c][f] == 1:
                    count += 1
    
    print count