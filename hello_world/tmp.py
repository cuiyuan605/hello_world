#coding=utf-8
import random

#map
class tmp():
    def __init__(self):
        pass
row=tmp()
row.career=""
row.ad_id="100016"
row.tid={'4':'aaa','100010':'bbb','100017':'ccc'}
cat_ad_title_map={
    '':['111','222','333'],
    'calendar':['444','555','666'],
    'qimao':['777','888','999']
    }
#new var
ad_material_info_map={
    '100017':['2','3'],
    '100010':['23'],
    '4':['8']
    }
ad_name_map={
    '100017':'',
    '100010':'calendar',
    '4':'qimao'
    }
#----

#rec_title
rec_dict = {}
'''
material_list = ['2', '3']
rec100017 = '100017_' + random.sample(material_list, 1)[0] + '_' + \
    str(random.sample(cat_ad_title_map.get(
        row.career, cat_ad_title_map['']), 1)[0])
rec100010 = '100010_23_' + str(random.sample(cat_ad_title_map['calendar'], 1)[0])
rec4 = '4_8_' + str(random.sample(cat_ad_title_map['qimao'], 1)[0])
rec_dict['100017'] = rec100017
rec_dict['100010'] = rec100010
rec_dict['4'] = rec4
'''
for ad_id_key in ad_material_info_map.keys():
    rec_id=ad_id_key+'_' + random.sample(ad_material_info_map[ad_id_key], 1)[0] + '_' + \
        str(random.sample(cat_ad_title_map[ad_name_map[ad_id_key]], 1)[0])
    rec_dict[ad_id_key]=rec_id
#result
print("rec_dict:",rec_dict)
#----

#update_rec
ad_id=str(row.ad_id)
'''
material_list = ['2', '3']
rec100017 = '100017_' + random.sample(material_list, 1)[0] + '_' + \
    str(random.sample(cat_ad_title_map[''], 1)[0])
rec100010 = '100010_23_' + str(random.sample(cat_ad_title_map['calendar'], 1)[0])
rec4  ='4_8_' + str(random.sample(cat_ad_title_map['qimao'], 1)[0])
ad_id = str(row.ad_id)
if ad_id.startswith('100017'):
    if row.tid is not None and row.tid.get('100017') is not None:
        ad_id = str(row.tid.get('100017'))
    else:
        ad_id = rec100017
if ad_id.startswith('100010'):
    if row.tid is not None and row.tid.get('100010') is not None:
        ad_id = str(row.tid.get('100010'))
    else:
        ad_id = rec100010
if ad_id.startswith('4'):
    if row.tid is not None and row.tid.get('4') is not None:
        ad_id = str(row.tid.get('4'))
    else:
        ad_id = rec4
'''
for ad_id_key in ad_material_info_map.keys():
    rec_id=ad_id_key+'_' + random.sample(ad_material_info_map[ad_id_key], 1)[0] + '_' + \
        str(random.sample(cat_ad_title_map[ad_name_map[ad_id_key]], 1)[0])
    if ad_id.startswith(ad_id_key):
        if row.tid is not None and row.tid.get(ad_id_key) is not None:
            ad_id = str(row.tid.get(ad_id_key))
        else:
            ad_id = ad_id_key+'_' + random.sample(ad_material_info_map[ad_id_key], 1)[0] +\
                '_' + str(random.sample(cat_ad_title_map[ad_name_map[ad_id_key]], 1)[0])

#result
print("ad_id:",ad_id)