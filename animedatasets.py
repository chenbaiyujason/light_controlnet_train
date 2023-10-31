import json
import concurrent.futures
import requests

API = "https://datasets-server.huggingface.co/"

def query(url):
    headers = {"Authorization": "Bearer api_XXXXXXXXXXXXXXXXXXXXXXX"}
    response = requests.get(url, headers=headers)
    return response.json()

def query_data(offset):
    datarows = []
    print(f"offset: {offset}")
    data = query(API+f"rows?dataset=ioclab%2Fanimesfw&config=default&split=train&offset={offset}&limit=100&length=100")
    try:
        for i in data['rows']:
            datarows.append([i['row']['image']['src'], i['row']['tags']])
        with open(f'./datas/data{int(offset/100)}.json', 'w') as f:
            json.dump(datarows, f)
        return datarows
    except Exception as e:
        print(e)
        print(data)
        return datarows

if __name__ == '__main__':
    num_rows = 3969800
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:  # 这里使用10个线程
        futures = []
        for n in range(num_rows//100):
            offset = 100 * n
            print(f"offset: {offset}")
            futures.append(executor.submit(query_data, offset))
        datarows = []
        i=0
        for future in concurrent.futures.as_completed(futures):
            print(i)
            datarows.extend(future.result())
            #每个任务完成，都添加到这个json文件中
            i+=1
        with open(f'./dataall.json', 'w') as f:
            json.dump(datarows, f)
            
