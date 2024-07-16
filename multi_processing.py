from flask import Flask, request, jsonify
import os
import threading
import queue

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 创建线程池和请求队列
request_queue = queue.Queue()
thread_pool = []

def worker():
    while True:
        request_data = request_queue.get()
        process_request(request_data)
        request_queue.task_done()

def process_request(request_data):
    file_content = request_data['file_content']
    filename = request_data['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # 保存文件
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    print(f"File {filename} saved at {file_path}")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    request_data = {
        'filename': file.filename,
        'file_content': file.read()
    }
    request_queue.put(request_data)
    return jsonify({"message": "File uploaded and processing in background."})

# 启动多个工作线程
for i in range(5):
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    thread_pool.append(t)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
