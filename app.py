# 추가 구현 필요한 부분
# 1. 클라이언트로부터 public key 받아와서 ID와 함께 저장하는 부분 (V)
# 2. 서버가 반환한 참/거짓과 인덱스 받아서 인덱스가 맞는지 확인 후 최종 참/거짓 리턴하는 부분 (V)
# 3. dist 저장시 user_id 앞에 붙이기 (V)
# 4. 처리가 제대로 되지 않았는데 processed에 저장되는 경우 확인
# 5. API 명세서 작성 (V)
# 6. 로그인 기능
# 7. 동일 ID에 대해 Upload할 때 기존 파일이 있을 경우 처리 방법 -> DELETE 기능 추가 (V)
# 8. 하나의 퍼블릭 키만 가지고 계속 사용할 가능성 -> DELETE에 키 빼고 모두 삭제, 키와 Reg 빼고 모두 삭제 기능 추가

# 여러 데이터셋 대상으로 테스트하여 정확도 기록


from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tenseal as ts
import numpy as np
import base64
import random
import os
import json
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

# 디렉토리 존재 여부 확인
os.makedirs('uploads/registered', exist_ok=True)
os.makedirs('uploads/new', exist_ok=True)
os.makedirs('uploads/key', exist_ok=True)
os.makedirs('processed', exist_ok=True)
os.makedirs('processed/dist', exist_ok=True)
os.makedirs('processed/protocol_app', exist_ok=True)
os.makedirs('processed/index', exist_ok=True)


def write_data(file_name, data):
    if type(data) == bytes:
        data = base64.b64encode(data)
    with open(file_name, 'wb') as f:
        f.write(data)

def read_data(file_name):
    with open(file_name, 'rb') as f:
        data = f.read()
    return base64.b64decode(data)

def func(x):
    x2 = x * x
    x3 = -0.5 * x
    x3 *= x2
    x3 += 1.5 * x
    return x3

def calculate_dist(enc_v1_proto, enc_reg_proto, user_id):
    global func
    key_path = os.path.join('uploads/key', f'{user_id}_public.txt')
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Key file for user ID {user_id} not found at {key_path}")

    print("Reading context...")
    context = ts.context_from(read_data(key_path))
    print("Context read successfully.")

    print("Creating CKKS vectors...")
    enc_v1 = ts.lazy_ckks_vector_from(enc_v1_proto)
    enc_v1.link_context(context)
    enc_reg = ts.lazy_ckks_vector_from(enc_reg_proto)
    enc_reg.link_context(context)
    print("CKKS vectors created successfully.")

    threshold = 100
    reverse_max_possible = 1 / 300

    print("Calculating distance...")
    dist = enc_v1 - enc_reg
    dist *= dist
    dist = dist.matmul(np.ones((128, 128), dtype=np.float64))
    dist -= threshold
    dist *= reverse_max_possible
    dist = func(func(func(dist)))
    dist = -dist + 1
    dist *= 1 / 2

    idx = random.randint(0, 127)
    print(f"Random index: {idx}")
    mask = np.zeros(128, dtype=np.float64)
    mask[idx] = 1

    print("Multiplying vectors...")
    dist *= mask

    alpha = 0.61
    beta = 10.0

    # 지정된 index를 제외한 나머지 index에 대한 값은 감마 분포를 따르도록 설정
    err = np.random.gamma(alpha, 1/beta, size=128)
    while (err > 0.55).any():  # 값이 0.55를 넘어가면 다시 뽑습니다.
        err = np.random.gamma(alpha, 1/beta, size=128)

    err[idx] = 0
    dist += err

    dist_filename = os.path.join("processed/dist", f"{user_id}_dist.txt")
    write_data(dist_filename, dist.serialize())

    # 인덱스를 json 형식으로 저장
    index_filename = os.path.join("processed/index", f"{user_id}_index.json")
    with open(index_filename, 'w') as f:
        json.dump({'idx': str(idx)}, f, ensure_ascii=False)

    return dist.serialize()

@app.route('/')
def home():
    return "2024-01 Capstone Design Flask Server Running"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_type = request.form.get('type')
    user_id = request.form.get('user_id')

    if file_type not in ['register', 'compare', 'key']:
        return jsonify({"error": "Invalid file type"}), 400

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    if file:
        if file_type == 'register':
            save_path = os.path.join('uploads/registered', f"{user_id}_{file.filename}")
        elif file_type == 'compare':
            save_path = os.path.join('uploads/new', f"{user_id}_{file.filename}")
        elif file_type == 'key':
            save_path = os.path.join('uploads/key', f"{user_id}_public.txt")
        
        file.save(save_path)

        if file_type == 'compare':
            return process_file(user_id, f"{user_id}_{file.filename}")
        
        return jsonify({"message": "File uploaded successfully"}), 200

def process_file(user_id, new_filename):
    new_file_path = os.path.join('uploads/new', new_filename)

    if not os.path.exists(new_file_path):
        return jsonify({"error": "New file not found"}), 400

    # 주어진 user_id에 해당하는 등록된 파일을 로드
    registered_files = os.listdir('uploads/registered')
    registered_file_path = None
    for file in registered_files:
        if file.startswith(user_id + "_"):
            registered_file_path = os.path.join('uploads/registered', file)
            break

    if not registered_file_path:
        return jsonify({"error": f"No registered files found for user ID {user_id}"}), 400

    print(f"Registered file path: {registered_file_path}")
    print(f"New file path: {new_file_path}")

    enc_reg_proto = read_data(registered_file_path)
    enc_v1_proto = read_data(new_file_path)

    print("enc_reg_proto and enc_v1_proto read successfully.")

    if not enc_v1_proto or not enc_reg_proto:
        return jsonify({"error": "Invalid input data"}), 400

    dist = calculate_dist(enc_v1_proto, enc_reg_proto, user_id)

    # 결과를 저장
    original_filename = new_filename.split('_', 1)[-1]  # Remove the user_id prefix from the original filename
    result_filename = f"{user_id}_protocol_app_" + original_filename
    # result_filename = "protocol_app_" + os.path.basename(new_file_path)
    result_path = os.path.join('processed/protocol_app', result_filename)
    with open(result_path, 'wb') as f:
        f.write(base64.b64encode(dist).decode().encode())

    print(f"Result saved to: {result_path}")

    return send_file(result_path, as_attachment=True, mimetype='text/plain')

@app.route('/verify', methods=['POST'])
def verify_result():
    try:
        data = request.json
        user_id = data.get('user_id')
        received_idx = data.get('idx')

        app.logger.info(f"Received JSON request: {data}")

        # 저장된 인덱스 읽기
        index_file_path = os.path.join("processed/index", f"{user_id}_index.json")
        if not os.path.exists(index_file_path):
            return jsonify({"error": "Index file not found"}), 400
        
        with open(index_file_path, 'r') as f:
            stored_idx = json.load(f)['idx']

        if received_idx == stored_idx:
            return jsonify({"message": "Identical Verified. Enter Allowed."}), 200
        elif received_idx == "-1":
            return jsonify({"message": "Not identical Verified. Enter Disallowed."}), 200
        else:
            return jsonify({"error": "Untrusted Response. Enter Disallowed."}), 400
    
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500
    
@app.route('/download/<user_id>', methods=['GET'])
def download_file(user_id):
    filename = f"{user_id}_protocol_app_enc_v1.txt"
    file_path = os.path.join('processed/protocol_app', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
@app.route('/delete/<user_id>', methods=['DELETE'])
def delete_user_data(user_id):
    # 이 디렉토리에서 삭제할 파일들을 탐색
    directories = [
        'processed/dist',
        'processed/index',
        'processed/protocol_app',
        'uploads/key',
        'uploads/registered',
        'uploads/new'
    ]

    deleted_files = []
    # 각 디렉토리에서 해당되는 유저 ID의 파일들을 탐색 및 삭제
    # startswith로 탐색하기 때문에 유저 ID 자릿수에 유의
    for directory in directories:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.startswith(user_id):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_files.append(file_path)
                        print(f"Deleted: {file_path}")

    return jsonify({"message": f"All files for user_id {user_id} deleted successfully.", "deleted_files": deleted_files}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
