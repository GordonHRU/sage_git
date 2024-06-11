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
        self.source = ["為蟬科昆蟲黑蚱Cryptotympana pustulata Fabricius的若蟲羽化時脫落的皮殼。", 
                       "百合科植物川貝母Fritillaria cirrhosa D. Don的乾燥鱗莖。",
                       "山蟬Cicada flammata Dist. (焰螓蟬Tibicen flammatus (Dist.))的若蟲羽化時脫落的皮殼。", 
                       "紫葳科植物美洲凌霄Campsis radicans (L.) Seem.或凌霄Campsis grandiflora (Thunb.) K. Schum.的乾燥花。", 
                       "百合科植物平貝母Fritillaria ussuriensis Maxim.的乾燥鱗莖。", 
                       "未知", 
                       "茄科植物白花曼陀羅Datura metel L.和毛曼陀羅Datura inoxia Mill.的乾燥花。前者習稱「南洋金花」，後者為「北洋金花」。",]
        self.Traits = ["略呈橢圓形而彎曲，黃棕色至棕色，半透明，有光澤，口吻發達，上唇寬短，下唇伸長成管狀。有絲狀觸角1對，多已斷落，複眼凸出。額部先端凸出，足3對，被黃棕色細毛，背部兩旁具小翅2對，背面呈十字形裂開，裂口向內捲曲，腹部共9節，腹部鈍，尾端呈鈍三角狀。體輕，中空，易碎，氣微，味淡。", 
                       "松貝呈類圓錐形或近球形，高0.3~0.8cm，直徑0.3~0.9cm。表面類白色。外層鱗葉2瓣，大小懸殊，大瓣緊抱小瓣，未抱部分呈新月形，頂部閉合。內有類圓柱形、頂端稍尖的心芽和小鱗葉1~2枚。先端鈍圓或稍尖，底部平，微凹入，中心有一灰褐色的鱗莖盤，偶有殘存鬚根。質硬脆，斷面白色，富粉性。氣微，味微苦。", 
                       "略呈橢圓形，較瘦而平直，淺黃棕色，半透明，有光澤。口吻發達，上唇寬短，下唇伸長成管狀。有絲狀觸角1對，多已斷落，複眼凸出。額部先端凸出，背部兩旁具小翅2對，足3對，被黃棕色細毛，背面呈十字形裂開，裂口向內捲曲，腹部上端較窄，下部鈍，腹部共9節，尾端呈尖刺狀凸起。體輕，中空，易碎，氣微，味淡。", 
                       "美洲凌霄：黃褐色至棕褐色，多皺縮捲曲。完整花朵長6~7cm。萼筒長1.5~2cm，硬革質，先端5齒裂，裂片短三角狀，長約為萼筒的1/3，萼筒外無明顯的縱棱；花冠內表面具明顯的深棕色脈紋。\n\
凌霄花：多皺縮捲曲，黃褐色至棕褐色，完整花朵長4~5cm。萼筒鐘狀，長2~2.5cm，裂片5，裂至中部，萼筒基部至萼齒尖有5條縱棱。花冠先端5裂，裂片半圓形，下部聯合呈漏斗狀，表面可見細脈紋，內表面較明顯。雄蕊4，著生在花冠上，2長2短，花藥個字形，花柱1，柱頭扁平。氣清香，味微苦、酸。蒴果長如莢，頂端鈍。", 
                       "本品呈扁球形，高0.5~1cm，直徑0.6~2cm。表面乳白色或淡黃白色，外層鱗葉2瓣，肥厚，大小相近或一片稍大抱合，頂端略平或微凹入，常稍開裂；中央鱗片小。質堅實而脆，斷麵粉性。氣微，味苦。", 
                       "未知", 
                       "白花曼陀羅：完整者長9~15cm，蒴花兩性，花冠喇叭狀，五裂，裂片先端有短尖，唇形，有重瓣者；雄蕊5，全部發育，插生於花冠筒；柱頭棒狀。心皮2，2室；中軸胎座，胚珠多數。蒴果。花萼在果時近基部環狀斷裂，僅基部宿存。\n\
毛曼陀羅：完整者長9~11cm，花冠裂片先端呈三角形，兩花冠裂片間有短尖，柱頭呈戟狀。"]
        self.taste = ["甘，寒。",
                      "甘、苦，微寒。", 
                      "甘，寒。", 
                      "甘、酸，寒。",
                      "苦、甘，微寒。", 
                      "未知", 
                      "辛，溫，有毒。"]
        self.efficacy_list = ["1.清熱解毒\n\
作用：蟬蛻具有清熱解毒的作用，常用於治療風熱感冒、發熱頭痛、咽喉腫痛等症狀。\n\
應用：可以用於發熱性感冒、咽喉炎、扁桃體炎等疾病的治療。\n\
2.透疹\n\
作用：蟬蛻有助於透疹，是一種常用的疹透藥，能幫助疹子透發和消退。\n\
應用：適用於麻疹初期疹出不暢或者疹子透發不完全的情況。\n\
3.止痙\n\
作用：蟬蛻具有止痙作用，能夠緩解抽搐和痙攣，特別是用於小兒驚風。\n\
應用：適用於治療小兒高熱驚風、癲癇引起的抽搐等症狀。\n\
4.祛風止癢\n\
作用：蟬蛻可以祛風止癢，適用於皮膚瘙癢症、蕁麻疹等皮膚病。\n\
應用：可以用於各種原因引起的皮膚瘙癢，如濕疹、蕁麻疹等。",
"1.止咳化痰\n\
作用：松貝具有良好的止咳和化痰作用，常用於治療各種類型的咳嗽，包括痰多、痰黏不易咳出的症狀。\n\
應用：適用於風熱咳嗽、肺熱咳嗽、慢性支氣管炎等。\n\
2.潤肺清熱\n\
作用：松貝能夠潤肺清熱，特別適合肺熱引起的乾咳少痰、咽喉腫痛等症狀。\n\
應用：用於肺熱燥咳、肺癰、支氣管擴張等。\n\
3.消腫散結\n\
作用：松貝具有消腫散結的功效，可以用於瘰癧、癰腫、乳癰等疾病的治療。\n\
應用：適用於各種結核、腫瘤以及乳腺增生等。",
"1.清熱解毒\n\
作用：蟬蛻具有清熱解毒的作用，常用於治療風熱感冒、發熱頭痛、咽喉腫痛等症狀。\n\
應用：可以用於發熱性感冒、咽喉炎、扁桃體炎等疾病的治療。\n\
2.透疹\n\
作用：蟬蛻有助於透疹，是一種常用的疹透藥，能幫助疹子透發和消退。\n\
應用：適用於麻疹初期疹出不暢或者疹子透發不完全的情況。\n\
3.止痙\n\
作用：蟬蛻具有止痙作用，能夠緩解抽搐和痙攣，特別是用於小兒驚風。\n\
應用：適用於治療小兒高熱驚風、癲癇引起的抽搐等症狀。\n\
4.祛風止癢\n\
作用：蟬蛻可以祛風止癢，適用於皮膚瘙癢症、蕁麻疹等皮膚病。\n\
應用：可以用於各種原因引起的皮膚瘙癢，如濕疹、蕁麻疹等。",
"1.活血祛瘀\n\
作用：凌霄花具有活血祛瘀的作用，能夠促進血液循環，減少瘀血引起的疼痛和腫脹。\n\
應用：適用於治療血瘀引起的閉經、痛經、跌打損傷、瘀血腫痛等。\n\
2.涼血散風\n\
作用：凌霄花能夠涼血散風，特別適用於血熱風盛所致的皮膚病，如風疹、濕疹等。\n\
應用：用於皮膚風疹瘙癢、濕疹等症狀。\n\
3.清熱解毒\n\
作用：凌霄花具有清熱解毒的功效，能夠幫助減輕熱毒引起的症狀。\n\
應用：適用於熱毒引起的瘡癰、癤腫等。\n\
4.美容養顏\n\
作用：傳統認為凌霄花具有美容養顏的功效，能夠減少色斑和皮膚色素沉著。\n\
應用：適用於面部色斑、黃褐斑等美容需求。",
"1.潤肺止咳\n\
作用：平貝具有潤肺止咳的功效，能有效緩解咳嗽症狀。\n\
應用：適用於各種類型的咳嗽，特別是乾咳、痰少的情況。常用於治療支氣管炎、肺炎等呼吸系統疾病。\n\
2.化痰\n\
作用：平貝具有化痰的作用，能幫助排出痰液，緩解因痰多引起的呼吸不暢。\n\
應用：適用於痰多、痰黏難以咳出的症狀，常用於治療慢性支氣管炎、哮喘等。\n\
3.清熱散結\n\
作用：平貝具有清熱散結的作用，能有效緩解因熱毒引起的炎症和結節。\n\
應用：適用於肺熱咳嗽、痰熱咳嗽、咽喉腫痛等症狀。", 
"Unknown",
"1.止痛\n\
作用：洋金花含有多種生物鹼，其中一些具有鎮痛作用，可用於緩解疼痛。\n\
應用：可用於治療頭痛、牙痛、風濕痛等，但要注意其毒性，應小心使用。\n\
2.鎮靜安神\n\
作用：洋金花有時被用於治療神經系統紊亂，具有一定的鎮靜和安眠作用。\n\
應用：適用於治療焦慮、失眠等神經系統相關疾病，但應注意用量和毒性。\n\
3.擴張支氣管\n\
作用：在一些傳統醫學中，洋金花被認為能夠擴張支氣管，減輕呼吸困難。\n\
應用：可用於治療哮喘、支氣管炎等呼吸系統相關疾病，但毒性大，應謹慎使用。\n\
4.解痙作用\n\
作用：洋金花也被用於一些情況下的解痙，如痙攣性疼痛或癲癇等。\n\
應用：可用於治療痙攣性疼痛、癲癇等，但要注意用量和毒性。"]
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