def BienSoXeChoFileLenh(bienSoXe):
    #18256 -> 182.56
    return bienSoXe[:3] + "." + bienSoXe[3:]

def BienSoXeChoFileVtracking(bienSoXe):
    #18256 -> 62C-18256
    return "62C-" + bienSoXe

def BienSoXeChoFileEpass(bienSoXe):
    #18256 -> 62C18256
    tail = bienSoXe[-3:]
    return "62C" + bienSoXe[:-3] + tail

def BienSoXeDeLayDuLieuVtracking(bienSoXe):
    #182.56 -> 18256
    chuoi = bienSoXe
    return chuoi.replace(".", "")