#coding:utf-8
success_code=300000000
success_message="Success"

invalid_image_code=300000001
invalid_image_message="Image invalid"

#==========================mask and extract feature module, internal code============================
mask_detect_person_code = 300101001
mask_detect_person_message = "Mask: Person Detected"

features_predict_success_code = 300101002
features_predict_success_message="Features: Predict Success"

#===============================return to web client=============================================
mask_detect_no_person_code=300101010
mask_detect_no_person_message="Mask: No Person Detected"

mask_low_scores_code = 300101011
mask_low_scores_message = "Mask: Scores is lower than threshold"

mask_grpc_connect_error_code = 300101012
mask_grpc_connect_error_message = "Mask: GRPC Connect Failed"

mask_predict_result_error_code = 300101013
mask_predict_result_error_message="Mask: Predict Value Error"

features_grpc_connect_error_code = 300101014
features_grpc_connect_error_message="Features: GRPC Connect Failed"

features_predict_result_error_code = 300101015
features_predict_result_error_message="Features: Predict Value Error"

features_connect_and_predict_error_code = 300101016
features_connect_and_predict_error_message="Features: Code 300101014 and Code 300101015"

features_value_list_not_math_code=300101017
features_value_list_not_math_message="Features: Key and Label Not Match "


#nsfw or sfw
nsfw_predict_error_code = 300102001
nsfw_predict_error_message="NSFW: predict error"

nsfw_predict_value_invalid_code = 300102002
nsfw_predict_value_invalid_message="NSFW: predict value invalid"
#=========================ocr===============================
text_detect_box_code = 300103001
text_detect_box_message = "Text_detection: Text Box Detected"

text_detect_grpc_connect_error_code = 300103002
text_detect_grpc_connect_error_message = "Text_detection: GRPC Connect Failed"

text_detect_predict_points_error_code = 300103003
text_detect_predict_points_error_message = "Text_detection: Predict Value Error"

ocr_grpc_connect_error_code = 300103004
ocr_grpc_connect_error_message = "Text_detection: GRPC Connect Failed"

ocr_predict_text_error_code = 300103005
ocr_predict_text_error_message = "Text_detection: Predict Value Error"
#===============================facenet============================================
face_detect_box_code = 300104001
face_detect_box_message = "Face_detection: Face Detected"

no_face_detect_code = 300104002
no_face_detect_message = "Face_detection: No Face Detected"

source_images_invalid_error_code = 300104003
source_images_invalid_error_message = "Face_detection: Source Image Invalid"

face_detect_grpc_connect_error_code = 300104004
face_detect_grpc_connect_error_message = "Face_detection: GRPC Connect Failed"

face_detect_predict_points_error_code = 300104005
face_detect_predict_points_error_message = "Face_detection: Predict Value Error"

facenet_images_invalid_error_code = 300104006
facenet_images_invalid_error_message = "Face_detection: Face Image Invalid"

facenet_predict_success_code = 300104007
facenet_predict_success_message = "FaceNet: Predict Success"

facenet_grpc_connect_error_code = 300104008
facenet_grpc_connect_error_message = "FaceNet: GRPC Connect Failed"

facenet_image_grpc_error_code = 300104007
facenet_image_grpc__error_message = "FaceNet: Face Image Invalid or GRPC Connect Failed"
