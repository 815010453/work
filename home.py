import json
import os
import uvicorn
from language_graph import op_graph_file
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
import pydantic
from pydantic import BaseModel
import difflib

area = "四川省"


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


async def sichuan_map_data(years: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                           types: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                           history=Body([],
                                        description="历史对话",
                                        examples=[[
                                            {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                            {"role": "assistant", "content": "虎头虎脑"}]]
                                        ),
                           stream: bool = Body(False, description="流式输出"),
                           ):
    global area
    question = ["2020",
                "2021",
                "2022",
                "2023"]
    # 字符串相似度匹配
    match = difflib.get_close_matches(years, question, n=1, cutoff=0.8)
    if not match:
        config_graph = {"backgroundColor": 'transparent'}
        return ChatMessage(data=str(config_graph))
    match = match[0]
    if types == "地震":
        if match == question[0]:
            year = "2020"  # 年份
            topic = '地震'  # 主题
            unit = "次"  # 单位
            area = "四川省"  # 地区
            # 数据
            mapdata_2020 = [
                ['地区', "2020"],
                ['成都市', "3"],
                ['绵阳市', "12"],
                ['自贡市', "8"],
                ['泸州市', "1"],
                ['德阳市', "2"],
                ['广元市', "4"],
                ['遂宁市', "0"],
                ['内江市', "5"],
                ['乐山市', "3"],
                ['资阳市', "0"],
                ['南充市', "0"],
                ['达州市', "0"],
                ['雅安市', "1"],
                ['广安市', "0"],
                ['巴中市', "0"],
                ['眉山市', "0"],
                ['宜宾市', "42"],
                ['攀枝花市', "0"],
                ['阿坝藏族羌族自治州', "7"],
                ['甘孜藏族自治州', "13"],
                ['凉山彝族自治州', "3"]
            ]
            if os.path.exists("language_graph/sichuan_2023dizhen.json"):
                with open("language_graph/sichuan_2023dizhen.json", encoding='utf-8') as f:
                    graph = eval(f.read())
                    categories_data = [graph["categories"][0]["name"]]
                    config_graph = {
                        "backgroundColor": 'transparent',
                        "title": {
                            "text": year + "年" + area + topic + "数据",
                            "subtext": 'Natural disaster situation',
                            "sublink": 'https://www.mem.gov.cn/xw/yjglbgzdt/202307/t20230707_455762.shtml',
                            "top": 'top',
                            "left": 'left'
                        },
                        "dataset": {
                            "source": mapdata_2020,
                        },
                        "visualMap": {
                            "type": "continuous", "min": 0, "max": 30, "text": ["高", "低"], "top": "center",
                            "inRange": {"color": ["mistyrose", "indianred", "orangered", "red"]},
                            "seriesIndex": 0
                        },
                        "toolbox": {
                            "show": 1,
                            "left": "left",
                            "top": "35%",
                            "feature": {
                                "restore": {}
                            }
                        },
                        "legend": [{
                            "data": categories_data
                        }],
                        "tooltip": {},
                        "animationDuration": 1500,
                        "animationEasingUpdate": 'quinticInOut',
                        "series": [
                            {
                                "type": "map",
                                "datasetIndex": 0,
                                "zoom": 1.1,
                                "name": year + area + topic + "数据",
                                "map": "Sichuan",
                                "roam": "scale",
                                "zlevel": 1,
                                "label": {"show": 0, "color": "transparent"},
                                "tooltip": {"trigger": "item", "valueFormatter": "{b}<br/>{c}(" + unit + ")",
                                            "formatter": "{b}<br/>{c}(" + unit + ")"},
                                "itemStyle": {
                                    "areaColor": '#fac858'
                                },
                                "showLegendSymbol": 0,
                                "emphasis": {
                                    "label": {
                                        "show": 0,
                                        "position": "inside"
                                    }
                                },
                                "select": {
                                    "label": {
                                        "show": 1
                                    }
                                },
                                # "silent": 1
                            },
                            {
                                "name": '四川省',
                                "zlevel": 100,
                                "type": 'graph',
                                # "visualMap": 0,
                                "layout": 'none',
                                "data": graph["nodes"],
                                "links": graph["links"],
                                "categories": graph["categories"],
                                "center": [103.265735, 31.259462],
                                "zoom": 0.68,
                                "roam": "scale",
                                "symbol": 'circle',
                                "label": {
                                    "show": 1,
                                    "position": 'inside',
                                    "formatter": '{b}',
                                    "color": "black"
                                },
                                "labelLayout": {
                                    "hideOverlap": 1
                                },
                                "scaleLimit": {
                                    "min": 0.4,
                                    "max": 4
                                },
                                "lineStyle": {
                                    "width": 5,
                                    "color": 'source',
                                    "curveness": 0.3
                                },
                                "emphasis": {
                                    "focus": 'adjacency',
                                    "lineStyle": {
                                        "width": 10
                                    }
                                }
                            }
                        ]
                    }
            return ChatMessage(data=str(config_graph))
        elif match == question[1]:
            year = "2021"  # 年份
            topic = '地震'  # 主题
            unit = "次"  # 单位
            area = "四川省"  # 地区
            # 数据
            mapdata_2021 = [
                ['地区', "2021"],
                ['成都市', "1"],
                ['绵阳市', "8"],
                ['自贡市', "4"],
                ['泸州市', "11"],
                ['德阳市', "0"],
                ['广元市', "3"],
                ['遂宁市', "0"],
                ['内江市', "6"],
                ['乐山市', "5"],
                ['资阳市', "0"],
                ['南充市', "0"],
                ['达州市', "0"],
                ['雅安市', "2"],
                ['广安市', "0"],
                ['巴中市', "0"],
                ['眉山市', "0"],
                ['宜宾市', "24"],
                ['攀枝花市', "0"],
                ['阿坝藏族羌族自治州', "12"],
                ['甘孜藏族自治州', "4"],
                ['凉山彝族自治州', "13"]
            ]
            if os.path.exists("language_graph/sichuan_2023dizhen.json"):
                with open("language_graph/sichuan_2023dizhen.json", encoding='utf-8') as f:
                    graph = eval(f.read())
                    categories_data = [graph["categories"][0]["name"]]
                    config_graph = {
                        "backgroundColor": 'transparent',
                        "title": {
                            "text": year + "年" + area + topic + "数据",
                            "subtext": 'Natural disaster situation',
                            "sublink": 'https://www.mem.gov.cn/xw/yjglbgzdt/202307/t20230707_455762.shtml',
                            "top": 'top',
                            "left": 'left'
                        },
                        "dataset": {
                            "source": mapdata_2021,
                        },
                        "visualMap": {
                            "type": "continuous", "min": 0, "max": 30, "text": ["高", "低"], "top": "center",
                            "inRange": {"color": ["mistyrose", "indianred", "orangered", "red"]},
                            "seriesIndex": 0
                        },
                        "toolbox": {
                            "show": 1,
                            "left": "left",
                            "top": "35%",
                            "feature": {
                                "restore": {}
                            }
                        },
                        "legend": [{
                            "data": categories_data
                        }],
                        "tooltip": {},
                        "animationDuration": 1500,
                        "animationEasingUpdate": 'quinticInOut',
                        "series": [
                            {
                                "type": "map",
                                "datasetIndex": 0,
                                "zoom": 1.1,
                                "name": year + area + topic + "数据",
                                "map": "Sichuan",
                                "roam": "scale",
                                "zlevel": 1,
                                "label": {"show": 0, "color": "transparent"},
                                "tooltip": {"trigger": "item", "valueFormatter": "{b}<br/>{c}(" + unit + ")",
                                            "formatter": "{b}<br/>{c}(" + unit + ")"},
                                "itemStyle": {
                                    "areaColor": '#fac858'
                                },
                                "showLegendSymbol": 0,
                                "emphasis": {
                                    "label": {
                                        "show": 0,
                                        "position": "inside"
                                    }
                                },
                                "select": {
                                    "label": {
                                        "show": 1
                                    }
                                },
                                # "silent": 1
                            },
                            {
                                "name": '四川省',
                                "zlevel": 100,
                                "type": 'graph',
                                # "visualMap": 0,
                                "layout": 'none',
                                "data": graph["nodes"],
                                "links": graph["links"],
                                "categories": graph["categories"],
                                "center": [103.265735, 31.259462],
                                "zoom": 0.68,
                                "roam": "scale",
                                "symbol": 'circle',
                                "label": {
                                    "show": 1,
                                    "position": 'inside',
                                    "formatter": '{b}',
                                    "color": "black"
                                },
                                "labelLayout": {
                                    "hideOverlap": 1
                                },
                                "scaleLimit": {
                                    "min": 0.4,
                                    "max": 4
                                },
                                "lineStyle": {
                                    "width": 5,
                                    "color": 'source',
                                    "curveness": 0.3
                                },
                                "emphasis": {
                                    "focus": 'adjacency',
                                    "lineStyle": {
                                        "width": 10
                                    }
                                }
                            }
                        ]
                    }
            return ChatMessage(data=str(config_graph))
        elif match == question[2]:
            year = "2022"  # 年份
            topic = '地震'  # 主题
            unit = "次"  # 单位
            area = "四川省"  # 地区
            # 数据
            mapdata_2022 = [
                ['地区', "2022"],
                ['成都市', "1"],
                ['绵阳市', "6"],
                ['自贡市', "0"],
                ['泸州市', "3"],
                ['德阳市', "0"],
                ['广元市', "2"],
                ['遂宁市', "0"],
                ['内江市', "0"],
                ['乐山市', "4"],
                ['资阳市', "0"],
                ['南充市', "0"],
                ['达州市', "0"],
                ['雅安市', "15"],
                ['广安市', "0"],
                ['巴中市', "0"],
                ['眉山市', "1"],
                ['宜宾市', "17"],
                ['攀枝花市', "0"],
                ['阿坝藏族羌族自治州', "23"],
                ['甘孜藏族自治州', "22"],
                ['凉山彝族自治州', "2"]
            ]
            if os.path.exists("language_graph/sichuan_2023dizhen.json"):
                with open("language_graph/sichuan_2023dizhen.json", encoding='utf-8') as f:
                    graph = eval(f.read())
                    categories_data = [graph["categories"][0]["name"]]
                    config_graph = {
                        "backgroundColor": 'transparent',
                        "title": {
                            "text": year + "年" + area + topic + "数据",
                            "subtext": 'Natural disaster situation',
                            "sublink": 'https://www.mem.gov.cn/xw/yjglbgzdt/202307/t20230707_455762.shtml',
                            "top": 'top',
                            "left": 'left'
                        },
                        "dataset": {
                            "source": mapdata_2022,
                        },
                        "visualMap": {
                            "type": "continuous", "min": 0, "max": 30, "text": ["高", "低"], "top": "center",
                            "inRange": {"color": ["mistyrose", "indianred", "orangered", "red"]},
                            "seriesIndex": 0
                        },
                        "toolbox": {
                            "show": 1,
                            "left": "left",
                            "top": "35%",
                            "feature": {
                                "restore": {}
                            }
                        },
                        "legend": [{
                            "data": categories_data
                        }],
                        "tooltip": {},
                        "animationDuration": 1500,
                        "animationEasingUpdate": 'quinticInOut',
                        "series": [
                            {
                                "type": "map",
                                "datasetIndex": 0,
                                "zoom": 1.1,
                                "name": year + area + topic + "数据",
                                "map": "Sichuan",
                                "roam": "scale",
                                "zlevel": 1,
                                "label": {"show": 0, "color": "transparent"},
                                "tooltip": {"trigger": "item", "valueFormatter": "{b}<br/>{c}(" + unit + ")",
                                            "formatter": "{b}<br/>{c}(" + unit + ")"},
                                "itemStyle": {
                                    "areaColor": '#fac858'
                                },
                                "showLegendSymbol": 0,
                                "emphasis": {
                                    "label": {
                                        "show": 0,
                                        "position": "inside"
                                    }
                                },
                                "select": {
                                    "label": {
                                        "show": 1
                                    }
                                },
                                # "silent": 1
                            },
                            {
                                "name": '四川省',
                                "zlevel": 100,
                                "type": 'graph',
                                # "visualMap": 0,
                                "layout": 'none',
                                "data": graph["nodes"],
                                "links": graph["links"],
                                "categories": graph["categories"],
                                "center": [103.265735, 31.259462],
                                "zoom": 0.68,
                                "roam": "scale",
                                "symbol": 'circle',
                                "label": {
                                    "show": 1,
                                    "position": 'inside',
                                    "formatter": '{b}',
                                    "color": "black"
                                },
                                "labelLayout": {
                                    "hideOverlap": 1
                                },
                                "scaleLimit": {
                                    "min": 0.4,
                                    "max": 4
                                },
                                "lineStyle": {
                                    "width": 5,
                                    "color": 'source',
                                    "curveness": 0.3
                                },
                                "emphasis": {
                                    "focus": 'adjacency',
                                    "lineStyle": {
                                        "width": 10
                                    }
                                }
                            }
                        ]
                    }
            return ChatMessage(data=str(config_graph))
        elif match == question[3]:
            year = "2023"  # 年份
            topic = '地震'  # 主题
            unit = "次"  # 单位
            area = "四川省"  # 地区
            # 数据
            mapdata_2023 = [
                ['地区', "2023"],
                ['成都市', "1"],
                ['绵阳市', "2"],
                ['自贡市', "1"],
                ['泸州市', "2"],
                ['德阳市', "0"],
                ['广元市', "0"],
                ['遂宁市', "0"],
                ['内江市', "9"],
                ['乐山市', "1"],
                ['资阳市', "0"],
                ['南充市', "0"],
                ['达州市', "0"],
                ['雅安市', "5"],
                ['广安市', "0"],
                ['巴中市', "0"],
                ['眉山市', "1"],
                ['宜宾市', "20"],
                ['攀枝花市', "0"],
                ['阿坝藏族羌族自治州', "1"],
                ['甘孜藏族自治州', "24"],
                ['凉山彝族自治州', "2"],
            ]
            if os.path.exists("language_graph/sichuan_2023dizhen.json"):
                with open("language_graph/sichuan_2023dizhen.json", encoding='utf-8') as f:
                    graph = eval(f.read())
                    categories_data = [graph["categories"][0]["name"]]
                    config_graph = {
                        "backgroundColor": 'transparent',
                        "title": {
                            "text": year + "年" + area + topic + "数据",
                            "subtext": 'Natural disaster situation',
                            "sublink": 'https://www.mem.gov.cn/xw/yjglbgzdt/202307/t20230707_455762.shtml',
                            "top": 'top',
                            "left": 'left'
                        },
                        "dataset": {
                            "source": mapdata_2023,
                        },
                        "visualMap": {
                            "type": "continuous", "min": 0, "max": 30, "text": ["高", "低"], "top": "center",
                            "inRange": {"color": ["mistyrose", "indianred", "orangered", "red"]},
                            "seriesIndex": 0
                        },
                        "toolbox": {
                            "show": 1,
                            "left": "left",
                            "top": "35%",
                            "feature": {
                                "restore": {}
                            }
                        },
                        "legend": [{
                            "data": categories_data
                        }],
                        "tooltip": {},
                        "animationDuration": 1500,
                        "animationEasingUpdate": 'quinticInOut',
                        "series": [
                            {
                                "type": "map",
                                "datasetIndex": 0,
                                "zoom": 1.1,
                                "name": year + area + topic + "数据",
                                "map": "Sichuan",
                                "roam": "scale",
                                "zlevel": 1,
                                "label": {"show": 0, "color": "transparent"},
                                "tooltip": {"trigger": "item", "valueFormatter": "{b}<br/>{c}(" + unit + ")",
                                            "formatter": "{b}<br/>{c}(" + unit + ")"},
                                "itemStyle": {
                                    "areaColor": '#fac858'
                                },
                                "showLegendSymbol": 0,
                                "emphasis": {
                                    "label": {
                                        "show": 0,
                                        "position": "inside"
                                    }
                                },
                                "select": {
                                    "label": {
                                        "show": 1
                                    }
                                },
                                # "silent": 1
                            },
                            {
                                "name": '四川省',
                                "zlevel": 100,
                                "type": 'graph',
                                # "visualMap": 0,
                                "layout": 'none',
                                "data": graph["nodes"],
                                "links": graph["links"],
                                "categories": graph["categories"],
                                "center": [103.265735, 31.259462],
                                "zoom": 0.68,
                                "roam": "scale",
                                "symbol": 'circle',
                                "label": {
                                    "show": 1,
                                    "position": 'inside',
                                    "formatter": '{b}',
                                    "color": "black"
                                },
                                "labelLayout": {
                                    "hideOverlap": 1
                                },
                                "scaleLimit": {
                                    "min": 0.4,
                                    "max": 4
                                },
                                "lineStyle": {
                                    "width": 5,
                                    "color": 'source',
                                    "curveness": 0.3
                                },
                                "emphasis": {
                                    "focus": 'adjacency',
                                    "lineStyle": {
                                        "width": 10
                                    }
                                }
                            }
                        ]
                    }
            return ChatMessage(data=str(config_graph))
    elif types == "滑坡":
        if match == question[0]:
            year = "2020"  # 年份
            topic = '山体滑坡'  # 主题
            unit = "次"  # 单位
            area = "四川省"  # 地区
            # 数据
            mapdata_2020 = [
                ['地区', "2020"],
                ['成都市', "0"],
                ['绵阳市', "0"],
                ['自贡市', "0"],
                ['泸州市', "0"],
                ['德阳市', "1"],
                ['广元市', "0"],
                ['遂宁市', "0"],
                ['内江市', "0"],
                ['乐山市', "0"],
                ['资阳市', "0"],
                ['南充市', "0"],
                ['达州市', "0"],
                ['雅安市', "0"],
                ['广安市', "0"],
                ['巴中市', "0"],
                ['眉山市', "0"],
                ['宜宾市', "0"],
                ['攀枝花市', "0"],
                ['阿坝藏族羌族自治州', "0"],
                ['甘孜藏族自治州', "1"],
                ['凉山彝族自治州', "0"],
            ]
            if os.path.exists("language_graph/sichuan_2023huapo.json"):
                with open("language_graph/sichuan_2023huapo.json", encoding='utf-8') as f:
                    graph = eval(f.read())
                    categories_data = [graph["categories"][0]["name"]]
                    config_graph = {
                        "backgroundColor": 'transparent',
                        "title": {
                            "text": year + "年" + area + topic + "数据",
                            "subtext": 'Natural disaster situation',
                            "sublink": 'https://www.mem.gov.cn/xw/yjglbgzdt/202307/t20230707_455762.shtml',
                            "top": 'top',
                            "left": 'left'
                        },
                        "dataset": {
                            "source": mapdata_2020,
                        },
                        "visualMap": {
                            "type": "continuous", "min": 0, "max": 30, "text": ["高", "低"], "top": "center",
                            "inRange": {"color": ["mistyrose", "indianred", "orangered", "red"]},
                            "seriesIndex": 0
                        },
                        "toolbox": {
                            "show": 1,
                            "left": "left",
                            "top": "35%",
                            "feature": {
                                "restore": {}
                            }
                        },
                        "legend": [{
                            "data": categories_data
                        }],
                        "tooltip": {},
                        "animationDuration": 1500,
                        "animationEasingUpdate": 'quinticInOut',
                        "series": [
                            {
                                "type": "map",
                                "datasetIndex": 0,
                                "zoom": 1.1,
                                "name": year + area + topic + "数据",
                                "map": "Sichuan",
                                "roam": "scale",
                                "zlevel": 1,
                                "label": {"show": 0, "color": "transparent"},
                                "tooltip": {"trigger": "item", "valueFormatter": "{b}<br/>{c}(" + unit + ")",
                                            "formatter": "{b}<br/>{c}(" + unit + ")"},
                                "itemStyle": {
                                    "areaColor": '#fac858'
                                },
                                "showLegendSymbol": 0,
                                "emphasis": {
                                    "label": {
                                        "show": 0,
                                        "position": "inside"
                                    }
                                },
                                "select": {
                                    "label": {
                                        "show": 1
                                    }
                                },
                                # "silent": 1
                            },
                            {
                                "name": '四川省',
                                "zlevel": 100,
                                "type": 'graph',
                                # "visualMap": 0,
                                "layout": 'none',
                                "data": graph["nodes"],
                                "links": graph["links"],
                                "categories": graph["categories"],
                                "center": [103.265735, 31.259462],
                                "zoom": 0.68,
                                "roam": "scale",
                                "symbol": 'circle',
                                "label": {
                                    "show": 1,
                                    "position": 'inside',
                                    "formatter": '{b}',
                                    "color": "black"
                                },
                                "labelLayout": {
                                    "hideOverlap": 1
                                },
                                "scaleLimit": {
                                    "min": 0.4,
                                    "max": 4
                                },
                                "lineStyle": {
                                    "width": 5,
                                    "color": 'source',
                                    "curveness": 0.3
                                },
                                "emphasis": {
                                    "focus": 'adjacency',
                                    "lineStyle": {
                                        "width": 10
                                    }
                                }
                            }
                        ]
                    }
            return ChatMessage(data=str(config_graph))
        elif match == question[1]:
            year = "2021"  # 年份
            topic = '山体滑坡'  # 主题
            unit = "次"  # 单位
            area = "四川省"  # 地区
            # 数据
            mapdata_2021 = [
                ['地区', "2021"],
                ['成都市', "0"],
                ['绵阳市', "0"],
                ['自贡市', "0"],
                ['泸州市', "0"],
                ['德阳市', "0"],
                ['广元市', "0"],
                ['遂宁市', "1"],
                ['内江市', "0"],
                ['乐山市', "1"],
                ['资阳市', "0"],
                ['南充市', "2"],
                ['达州市', "1"],
                ['雅安市', "0"],
                ['广安市', "0"],
                ['巴中市', "0"],
                ['眉山市', "0"],
                ['宜宾市', "1"],
                ['攀枝花市', "0"],
                ['阿坝藏族羌族自治州', "0"],
                ['甘孜藏族自治州', "0"],
                ['凉山彝族自治州', "0"],
            ]
            if os.path.exists("language_graph/sichuan_2023huapo.json"):
                with open("language_graph/sichuan_2023huapo.json", encoding='utf-8') as f:
                    graph = eval(f.read())
                    categories_data = [graph["categories"][0]["name"]]
                    config_graph = {
                        "backgroundColor": 'transparent',
                        "title": {
                            "text": year + "年" + area + topic + "数据",
                            "subtext": 'Natural disaster situation',
                            "sublink": 'https://www.mem.gov.cn/xw/yjglbgzdt/202307/t20230707_455762.shtml',
                            "top": 'top',
                            "left": 'left'
                        },
                        "dataset": {
                            "source": mapdata_2021,
                        },
                        "visualMap": {
                            "type": "continuous", "min": 0, "max": 30, "text": ["高", "低"], "top": "center",
                            "inRange": {"color": ["mistyrose", "indianred", "orangered", "red"]},
                            "seriesIndex": 0
                        },
                        "toolbox": {
                            "show": 1,
                            "left": "left",
                            "top": "35%",
                            "feature": {
                                "restore": {}
                            }
                        },
                        "legend": [{
                            "data": categories_data
                        }],
                        "tooltip": {},
                        "animationDuration": 1500,
                        "animationEasingUpdate": 'quinticInOut',
                        "series": [
                            {
                                "type": "map",
                                "datasetIndex": 0,
                                "zoom": 1.1,
                                "name": year + area + topic + "数据",
                                "map": "Sichuan",
                                "roam": "scale",
                                "zlevel": 1,
                                "label": {"show": 0, "color": "transparent"},
                                "tooltip": {"trigger": "item", "valueFormatter": "{b}<br/>{c}(" + unit + ")",
                                            "formatter": "{b}<br/>{c}(" + unit + ")"},
                                "itemStyle": {
                                    "areaColor": '#fac858'
                                },
                                "showLegendSymbol": 0,
                                "emphasis": {
                                    "label": {
                                        "show": 0,
                                        "position": "inside"
                                    }
                                },
                                "select": {
                                    "label": {
                                        "show": 1
                                    }
                                },
                                # "silent": 1
                            },
                            {
                                "name": '四川省',
                                "zlevel": 100,
                                "type": 'graph',
                                # "visualMap": 0,
                                "layout": 'none',
                                "data": graph["nodes"],
                                "links": graph["links"],
                                "categories": graph["categories"],
                                "center": [103.265735, 31.259462],
                                "zoom": 0.68,
                                "roam": "scale",
                                "symbol": 'circle',
                                "label": {
                                    "show": 1,
                                    "position": 'inside',
                                    "formatter": '{b}',
                                    "color": "black"
                                },
                                "labelLayout": {
                                    "hideOverlap": 1
                                },
                                "scaleLimit": {
                                    "min": 0.4,
                                    "max": 4
                                },
                                "lineStyle": {
                                    "width": 5,
                                    "color": 'source',
                                    "curveness": 0.3
                                },
                                "emphasis": {
                                    "focus": 'adjacency',
                                    "lineStyle": {
                                        "width": 10
                                    }
                                }
                            }
                        ]
                    }
            return ChatMessage(data=str(config_graph))
        elif match == question[2]:
            year = "2022"  # 年份
            topic = '山体滑坡'  # 主题
            unit = "次"  # 单位
            area = "四川省"  # 地区
            # 数据
            mapdata_2022 = [
                ['地区', "2022"],
                ['成都市', "0"],
                ['绵阳市', "0"],
                ['自贡市', "0"],
                ['泸州市', "0"],
                ['德阳市', "0"],
                ['广元市', "0"],
                ['遂宁市', "0"],
                ['内江市', "0"],
                ['乐山市', "0"],
                ['资阳市', "0"],
                ['南充市', "0"],
                ['达州市', "0"],
                ['雅安市', "0"],
                ['广安市', "0"],
                ['巴中市', "0"],
                ['眉山市', "0"],
                ['宜宾市', "0"],
                ['攀枝花市', "0"],
                ['阿坝藏族羌族自治州', "0"],
                ['甘孜藏族自治州', "0"],
                ['凉山彝族自治州', "0"],
            ]
            if os.path.exists("language_graph/sichuan_2023huapo.json"):
                with open("language_graph/sichuan_2023huapo.json", encoding='utf-8') as f:
                    graph = eval(f.read())
                    categories_data = [graph["categories"][0]["name"]]
                    config_graph = {
                        "backgroundColor": 'transparent',
                        "title": {
                            "text": year + "年" + area + topic + "数据",
                            "subtext": 'Natural disaster situation',
                            "sublink": 'https://www.mem.gov.cn/xw/yjglbgzdt/202307/t20230707_455762.shtml',
                            "top": 'top',
                            "left": 'left'
                        },
                        "dataset": {
                            "source": mapdata_2022,
                        },
                        "visualMap": {
                            "type": "continuous", "min": 0, "max": 30, "text": ["高", "低"], "top": "center",
                            "inRange": {"color": ["mistyrose", "indianred", "orangered", "red"]},
                            "seriesIndex": 0
                        },
                        "toolbox": {
                            "show": 1,
                            "left": "left",
                            "top": "35%",
                            "feature": {
                                "restore": {}
                            }
                        },
                        "legend": [{
                            "data": categories_data
                        }],
                        "tooltip": {},
                        "animationDuration": 1500,
                        "animationEasingUpdate": 'quinticInOut',
                        "series": [
                            {
                                "type": "map",
                                "datasetIndex": 0,
                                "zoom": 1.1,
                                "name": year + area + topic + "数据",
                                "map": "Sichuan",
                                "roam": "scale",
                                "zlevel": 1,
                                "label": {"show": 0, "color": "transparent"},
                                "tooltip": {"trigger": "item", "valueFormatter": "{b}<br/>{c}(" + unit + ")",
                                            "formatter": "{b}<br/>{c}(" + unit + ")"},
                                "itemStyle": {
                                    "areaColor": '#fac858'
                                },
                                "showLegendSymbol": 0,
                                "emphasis": {
                                    "label": {
                                        "show": 0,
                                        "position": "inside"
                                    }
                                },
                                "select": {
                                    "label": {
                                        "show": 1
                                    }
                                },
                                # "silent": 1
                            },
                            {
                                "name": '四川省',
                                "zlevel": 100,
                                "type": 'graph',
                                # "visualMap": 0,
                                "layout": 'none',
                                "data": graph["nodes"],
                                "links": graph["links"],
                                "categories": graph["categories"],
                                "center": [103.265735, 31.259462],
                                "zoom": 0.68,
                                "roam": "scale",
                                "symbol": 'circle',
                                "label": {
                                    "show": 1,
                                    "position": 'inside',
                                    "formatter": '{b}',
                                    "color": "black"
                                },
                                "labelLayout": {
                                    "hideOverlap": 1
                                },
                                "scaleLimit": {
                                    "min": 0.4,
                                    "max": 4
                                },
                                "lineStyle": {
                                    "width": 5,
                                    "color": 'source',
                                    "curveness": 0.3
                                },
                                "emphasis": {
                                    "focus": 'adjacency',
                                    "lineStyle": {
                                        "width": 10
                                    }
                                }
                            }
                        ]
                    }
            return ChatMessage(data=str(config_graph))
        elif match == question[3]:
            year = "2023"  # 年份
            topic = '山体滑坡'  # 主题
            unit = "次"  # 单位
            area = "四川省"  # 地区
            # 数据
            mapdata_2023 = [
                ['地区', "2023"],
                ['成都市', "0"],
                ['绵阳市', "0"],
                ['自贡市', "0"],
                ['泸州市', "0"],
                ['德阳市', "0"],
                ['广元市', "0"],
                ['遂宁市', "0"],
                ['内江市', "0"],
                ['乐山市', "1"],
                ['资阳市', "0"],
                ['南充市', "0"],
                ['达州市', "1"],
                ['雅安市', "0"],
                ['广安市', "0"],
                ['巴中市', "0"],
                ['眉山市', "0"],
                ['宜宾市', "0"],
                ['攀枝花市', "0"],
                ['阿坝藏族羌族自治州', "1"],
                ['甘孜藏族自治州', "0"],
                ['凉山彝族自治州', "0"],
            ]
            if os.path.exists("language_graph/sichuan_2023huapo.json"):
                with open("language_graph/sichuan_2023huapo.json", encoding='utf-8') as f:
                    graph = eval(f.read())
                    categories_data = [graph["categories"][0]["name"]]
                    config_graph = {
                        "backgroundColor": 'transparent',
                        "title": {
                            "text": year + "年" + area + topic + "数据",
                            "subtext": 'Natural disaster situation',
                            "sublink": 'https://www.mem.gov.cn/xw/yjglbgzdt/202307/t20230707_455762.shtml',
                            "top": 'top',
                            "left": 'left'
                        },
                        "dataset": {
                            "source": mapdata_2023,
                        },
                        "visualMap": {
                            "type": "continuous", "min": 0, "max": 30, "text": ["高", "低"], "top": "center",
                            "inRange": {"color": ["mistyrose", "indianred", "orangered", "red"]},
                            "seriesIndex": 0
                        },
                        "toolbox": {
                            "show": 1,
                            "left": "left",
                            "top": "35%",
                            "feature": {
                                "restore": {}
                            }
                        },
                        "legend": [{
                            "data": categories_data
                        }],
                        "tooltip": {},
                        "animationDuration": 1500,
                        "animationEasingUpdate": 'quinticInOut',
                        "series": [
                            {
                                "type": "map",
                                "datasetIndex": 0,
                                "zoom": 1.1,
                                "name": year + area + topic + "数据",
                                "map": "Sichuan",
                                "roam": "scale",
                                "zlevel": 1,
                                "label": {"show": 0, "color": "transparent"},
                                "tooltip": {"trigger": "item", "valueFormatter": "{b}<br/>{c}(" + unit + ")",
                                            "formatter": "{b}<br/>{c}(" + unit + ")"},
                                "itemStyle": {
                                    "areaColor": '#fac858'
                                },
                                "showLegendSymbol": 0,
                                "emphasis": {
                                    "label": {
                                        "show": 0,
                                        "position": "inside"
                                    }
                                },
                                "select": {
                                    "label": {
                                        "show": 1
                                    }
                                },
                                # "silent": 1
                            },
                            {
                                "name": '四川省',
                                "zlevel": 100,
                                "type": 'graph',
                                # "visualMap": 0,
                                "layout": 'none',
                                "data": graph["nodes"],
                                "links": graph["links"],
                                "categories": graph["categories"],
                                "center": [103.265735, 31.259462],
                                "zoom": 0.68,
                                "roam": "scale",
                                "symbol": 'circle',
                                "label": {
                                    "show": 1,
                                    "position": 'inside',
                                    "formatter": '{b}',
                                    "color": "black"
                                },
                                "labelLayout": {
                                    "hideOverlap": 1
                                },
                                "scaleLimit": {
                                    "min": 0.4,
                                    "max": 4
                                },
                                "lineStyle": {
                                    "width": 5,
                                    "color": 'source',
                                    "curveness": 0.3
                                },
                                "emphasis": {
                                    "focus": 'adjacency',
                                    "lineStyle": {
                                        "width": 10
                                    }
                                }
                            }
                        ]
                    }
            return ChatMessage(data=str(config_graph))
    else:
        config_graph = {"backgroundColor": 'transparent'}
        return ChatMessage(data=str(config_graph))


async def OP_JSON(input_file: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                  query: str = Body(..., description="用户输入", examples=["恼羞成怒"])):
    if input_file and query:
        if input_file == "地震":
            if os.path.exists("language_graph/sichuan_2023dizhen.json"):
                info = {"tpl": op_graph_file.Op_json("language_graph/sichuan_2023dizhen.json", query)}
                return ResponseMessage(data=info)
            else:
                info = {"tpl": ""}
                return ResponseMessage(data=info)
        elif input_file == "滑坡":
            if os.path.exists("language_graph/sichuan_2023huapo.json"):
                info = {"tpl": op_graph_file.Op_json("language_graph/sichuan_2023huapo.json", query)}
                return ResponseMessage(data=info)
            else:
                info = {"tpl": ""}
                return ResponseMessage(data=info)
        else:
            info = {"tpl": ""}
            return ResponseMessage(data=info)
    else:
        info = {"tpl": ""}
    return ResponseMessage(data=info)


async def login(name: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                password: str = Body(..., description=123, examples=["恼羞成怒"]),
                code: str = Body(..., description="用户输入", examples=["恼羞成怒"])):
    if name == "yyh" and password == "123" and code == "4192":
        url = {"tpl": "连接成功!", "url": "http://127.0.0.1:5501/index_1.html#/home_page"}
    else:
        url = {"tpl": "连接失败!", "url": ""}
    return ResponseMessage(data=url)


async def sichuan_map_geojson():
    # 你看看你的China.json文件在哪 改一下路径
    if os.path.exists('Sichuan.json'):
        with open('Sichuan.json', 'r', encoding='utf-8') as f:
            js = json.load(f)
    else:
        js = {}
    return MapMessage(data=js)


if __name__ == '__main__':
    app = FastAPI(
        title="Test Server",
    )
    # 自己看看map_geojson函数里的路径！！！！！！
    app.get("/sichuan_map_geojson",
            tags=["get"],
            summary="获取geoJson文件")(sichuan_map_geojson)

    app.post("/login",
             tags=["post"],
             summary="获取geoJson文件")(login)

    app.post("/sichuan_map",
             tags=["post"],
             summary="地图的config")(sichuan_map_data)

    app.post("/graph_json",
             tags=["post"],
             summary="地图的config")(OP_JSON)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(app, host="127.0.0.1", port=7862)
