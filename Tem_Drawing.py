import json
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
import pydantic
from pydantic import BaseModel

area = "香港"


class BaseResponse(BaseModel):
    status: int = pydantic.Field(0, description="HTTP status code")
    msg: str = pydantic.Field("success", description="HTTP status message")

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "code": 0,
                "msg": "success",
            }
        }


class ChatMessage(BaseResponse):
    data: str = pydantic.Field(..., description="Map config")

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "status": 0,
                "msg": "success",
                "data": {}
            }
        }


class MapMessage(BaseResponse):
    data: dict = pydantic.Field(..., description="Map config")

    class Config:
        schema_extra = {
            "example": {
                "status": 0,
                "msg": "success",
                "data": {}
            }
        }


class ResponseMessage(BaseResponse):
    data: dict = pydantic.Field(..., description="Map config")

    class Config:
        schema_extra = {
            "example": {
                "status": 0,
                # "msg": "success",
                "data": {}
            }
        }


async def Tem_Draw(input_list: dict = Body(..., description="用户输入")):
    print(input_list)
    data_file_name = input_list["input_list"]['data_file']
    name_file_name = input_list["input_list"]['name_file']
    title_name = input_list["input_list"]['title']
    little_title_name = input_list["input_list"]['little_title']
    tool_name = input_list["input_list"]['tool']
    label_name = input_list["input_list"]['label']
    zoom_name = input_list["input_list"]['zoom']
    zoom_min = input_list["input_list"]['zoom_min']
    zoom_max = input_list["input_list"]['zoom_max']
    bgcolor_name = input_list["input_list"]['bgcolor']
    legend_name = input_list["input_list"]['legend']
    legend_color = legend_name.split(",")
    if tool_name == "true":
        tool_value = 1
    else:
        tool_value = 0
    if label_name == "true":
        label_value = 1
    else:
        label_value = 0
    if zoom_name == "true":
        zoom_name = "scale"
    if data_file_name:
        if os.path.exists("language_graph/" + data_file_name):
            with open("language_graph/" + data_file_name, encoding='utf-8') as f_data:
                data_file = eval(f_data.read())
    else:
        option = {}
        return ChatMessage(data=str(option))
    if name_file_name:
        if os.path.exists("language_graph/" + name_file_name):
            with open("language_graph/" + name_file_name, encoding='utf-8') as f_name:
                name_file = eval(f_name.read())
    else:
        name_file = {}
    option = {
        "backgroundColor": bgcolor_name,
        "title": {
            "text": title_name,
            "subtext": little_title_name
        },
        "tooltip": {
            "trigger": 'item',
            "formatter": '{b}<br/>{c} (p / km2)'
        },
        "toolbox": {
            "show": tool_value,
            "orient": 'vertical',
            "left": 'right',
            "top": 'center',
            "feature": {
                "dataView": {"readOnly": 0},
                "restore": {},
                "saveAsImage": {}
            }
        },
        "visualMap": {
            "min": 800,
            "max": 50000,
            "text": ['High', 'Low'],
            'realtime': 0,
            "calculable": 1,
            "inRange": {
                "color": legend_color
            }
        },
        "dataset": [{"source": data_file, "sourceHeader": 1}],
        "series": [
            {
                "type": "map",
                "name": '香港18区人口密度',
                "map": "HK",
                "roam": zoom_name,
                "datasetIndex": 0,
                "label": {
                    "show": label_value
                },
                "nameMap": name_file,
                "scaleLimit": {
                    "min": zoom_min,
                    "max": zoom_max
                },
            }
        ]
    }
    return ChatMessage(data=str(option))


async def map_geojson_HK():
    if os.path.exists('E:/YYH/pycharm/langchain-ChatGLM/language_graph/HK.json'):
        with open('E:/YYH/pycharm/langchain-ChatGLM/language_graph/HK.json', 'r', encoding='utf-8') as f:
            js = json.load(f)
    else:
        js = {}
    return MapMessage(data=js)


if __name__ == '__main__':
    app = FastAPI(
        title="Test Server",
    )
    app.get("/HK_geojson",
            tags=["get"],
            summary="获取geoJson文件")(map_geojson_HK)

    app.post("/Tem_Draw",
             tags=["post"],
             summary="地图的config")(Tem_Draw)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(app, host="127.0.0.1", port=7863)
