import urllib.request, json
def getJson():
    with urllib.request.urlopen("http://127.0.0.1/get-intent") as url:
        data = json.loads(url.read().decode())
        return data

# def testfunc():
    # print((getJson()))

# testfunc()