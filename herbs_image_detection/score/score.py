import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
import logging
import base64
import numpy as np

class predict_data():
    def __init__(self, iamge_data_dic):
        self.class_name_chinese = ['蟬蛻', '松貝', '金蟬蛻', '凌霄花', '平貝', '未知', '洋金花']
        self.class_name_Scientific = ['Cicadae Periostracum', 'Fritillariae Cirrhosae Bulbus', 'Cicadae Periostracum Flammatae', 'Campsis Flos', 'Fritillariae Ussuriensis Bulbus', 'Unknown', 'Daturae Flos']
        self.class_name_category = ['動物類','根及根莖類','動物類','花類','根及根莖類', 'Unknown','花類']
        self.class_source = ["為蟬科昆蟲黑蚱Cryptotympana pustulata Fabricius的若蟲羽化時脫落的皮殼。", 
                       "百合科植物川貝母Fritillaria cirrhosa D. Don的乾燥鱗莖。",
                       "山蟬Cicada flammata Dist. (焰螓蟬Tibicen flammatus (Dist.))的若蟲羽化時脫落的皮殼。", 
                       "紫葳科植物美洲凌霄Campsis radicans (L.) Seem.或凌霄Campsis grandiflora (Thunb.) K. Schum.的乾燥花。", 
                       "百合科植物平貝母Fritillaria ussuriensis Maxim.的乾燥鱗莖。", 
                       "未知", 
                       "茄科植物白花曼陀羅Datura metel L.和毛曼陀羅Datura inoxia Mill.的乾燥花。前者習稱「南洋金花」，後者為「北洋金花」。",]
        self.class_Traits = ["略呈橢圓形而彎曲，黃棕色至棕色，半透明，有光澤，口吻發達，上唇寬短，下唇伸長成管狀。有絲狀觸角1對，多已斷落，複眼凸出。額部先端凸出，足3對，被黃棕色細毛，背部兩旁具小翅2對，背面呈十字形裂開，裂口向內捲曲，腹部共9節，腹部鈍，尾端呈鈍三角狀。體輕，中空，易碎，氣微，味淡。", 
                       "松貝呈類圓錐形或近球形，高0.3~0.8cm，直徑0.3~0.9cm。表面類白色。外層鱗葉2瓣，大小懸殊，大瓣緊抱小瓣，未抱部分呈新月形，頂部閉合。內有類圓柱形、頂端稍尖的心芽和小鱗葉1~2枚。先端鈍圓或稍尖，底部平，微凹入，中心有一灰褐色的鱗莖盤，偶有殘存鬚根。質硬脆，斷面白色，富粉性。氣微，味微苦。", 
                       "略呈橢圓形，較瘦而平直，淺黃棕色，半透明，有光澤。口吻發達，上唇寬短，下唇伸長成管狀。有絲狀觸角1對，多已斷落，複眼凸出。額部先端凸出，背部兩旁具小翅2對，足3對，被黃棕色細毛，背面呈十字形裂開，裂口向內捲曲，腹部上端較窄，下部鈍，腹部共9節，尾端呈尖刺狀凸起。體輕，中空，易碎，氣微，味淡。", 
                       "美洲凌霄：黃褐色至棕褐色，多皺縮捲曲。完整花朵長6~7cm。萼筒長1.5~2cm，硬革質，先端5齒裂，裂片短三角狀，長約為萼筒的1/3，萼筒外無明顯的縱棱；花冠內表面具明顯的深棕色脈紋。\n\
凌霄花：多皺縮捲曲，黃褐色至棕褐色，完整花朵長4~5cm。萼筒鐘狀，長2~2.5cm，裂片5，裂至中部，萼筒基部至萼齒尖有5條縱棱。花冠先端5裂，裂片半圓形，下部聯合呈漏斗狀，表面可見細脈紋，內表面較明顯。雄蕊4，著生在花冠上，2長2短，花藥個字形，花柱1，柱頭扁平。氣清香，味微苦、酸。蒴果長如莢，頂端鈍。", 
                       "本品呈扁球形，高0.5~1cm，直徑0.6~2cm。表面乳白色或淡黃白色，外層鱗葉2瓣，肥厚，大小相近或一片稍大抱合，頂端略平或微凹入，常稍開裂；中央鱗片小。質堅實而脆，斷麵粉性。氣微，味苦。", 
                       "未知", 
                       "白花曼陀羅：完整者長9~15cm，蒴花兩性，花冠喇叭狀，五裂，裂片先端有短尖，唇形，有重瓣者；雄蕊5，全部發育，插生於花冠筒；柱頭棒狀。心皮2，2室；中軸胎座，胚珠多數。蒴果。花萼在果時近基部環狀斷裂，僅基部宿存。\n\
毛曼陀羅：完整者長9~11cm，花冠裂片先端呈三角形，兩花冠裂片間有短尖，柱頭呈戟狀。"]
        self.class_taste = ["甘，寒。",
                      "甘、苦，微寒。", 
                      "甘，寒。", 
                      "甘、酸，寒。",
                      "苦、甘，微寒。", 
                      "未知", 
                      "辛，溫，有毒。"]
        self.efficacy_list = ["散風除熱，利咽，透疹，退翳，解痙。用於風熱感冒，咽痛，音啞，麻疹不透，風疹瘙癢，目赤翳障，驚風抽搐，破傷風。", 
                              "清熱潤肺，化痰止咳。用於肺熱燥咳，乾咳少痰，陰虛勞嗽，咯痰帶血。", 
                              "散風除熱，利咽，透疹，退翳，解痙。用於風熱感冒，咽痛，音啞，麻疹不透，風疹瘙癢，目赤翳障，驚風抽搐，破傷風。", 
                              "行血去瘀，涼血祛風。用於經閉癓瘕，產後乳腫，風疹發紅，皮膚瘙癢，痤瘡。", 
                              "清熱潤肺，化痰止咳。主治主肺熱燥咳，乾咳少痰，陰虛勞嗽，咯痰帶血，瘰鬁，乳癰等。", 
                              "未知", 
                              "鎮痙、鎮靜、鎮痛、麻醉的功能。"]
        self.array = model.predict(iamge_data_dic)[0]
        self.value = max(self.array)
        self.index = np.argmax(self.array)
    def chinese_name(self):
        if self.value >= 0.5:
            return self.class_name_chinese[self.index]
        else:
            return '無法辨識'
    def Scientific_Name(self):
        if self.value >= 0.5:
            return self.class_name_Scientific[self.index]
        else:
            return 'Unknown'
    def category(self):
        if self.value >= 0.5:
            return self.class_name_category[self.index]
        else:
            return 'Unknown'
    def confidence(self):
        if self.value >= 0.5:
            return str(self.value * 100)
        else:
            return 'Unknown'
    def efficacy(self):
        if self.value >= 0.5:
            return self.efficacy_list[self.index]
        else:
            return 'Unknown'
    def source(self):
        if self.value >= 0.5:
            return self.class_source[self.index]
        else:
            return 'Unknown'        
    def Traits(self):
        if self.value >= 0.5:
            return self.class_Traits[self.index]
        else:
            return 'Unknown'    
    def taste(self):
        if self.value >= 0.5:
            return self.class_taste[self.index]
        else:
            return 'Unknown'       

def init():
    global model
    global img_height
    global img_width

    # model_path = "D:/code/git_Andy/sage_git/herbs_image_detection/model_herbs_D529_V2/model.keras"
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model.keras"
    )

    img_height = 224
    img_width = 224
    model = tf.keras.models.load_model(model_path)
    logging.error("end model load")

def run(raw_data):

    logging.error("start process")
    logging.error("input length = "+str(len(raw_data)))

    os.makedirs("score/picture", exist_ok=True)
    byte_data = bytes(raw_data, 'utf-8')
    with open("score/picture/image_base64.jpg", "wb") as fh:
        fh.write(base64.decodebytes(byte_data))
    image_path_dic = ("score")
    iamge_data_dic = tf.keras.preprocessing.image_dataset_from_directory(
        image_path_dic,
        image_size=(img_height, img_width),
        validation_split=None,
    )
    result_data  = predict_data(iamge_data_dic)

    logging.error("end process")
    return {
            "Confidence":result_data.confidence(),
            "Chinese name":result_data.chinese_name(),
            "Scientific Name":result_data.Scientific_Name(),
            "Category":result_data.category(),
            "Source":result_data.source(),
            "Traits":result_data.Traits(),
            "Taste":result_data.taste(),
            "efficacy":result_data.efficacy()
            }

# if __name__ == "__main__":
#     test_path = os.path.join('D:/code/git_Andy/sage_git/herbs_image_detection/score/test_pic/Chuanbeimu.jpg')
#     image =  Image.open(test_path).convert('RGB')
#     buffer = BytesIO()
#     image.save(buffer, "JPEG", quality=90)
#     img_str = base64.b64encode(buffer.getvalue())
#     image.close()
#     init()
#     print(run(str(img_str)))