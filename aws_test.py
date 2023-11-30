# app.py
from flask import Flask, request, jsonify
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from werkzeug.utils import secure_filename
import boto3
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET, REST_API_KEY




app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

class UploadForm(FlaskForm):
    image = FileField('Image', validators=[FileRequired(), FileAllowed(['jpg', 'png', 'jpeg'], 'Images only!')])

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)


@app.route('/', methods=['POST'])
def upload_image():
    try:
        # 이미지 파일 가져오기
        image = request.files['image']

        # 폴더 경로 지정
        folder_path = 'EditPage/Flask_img/'     
        filename = secure_filename(image.filename)
        key = folder_path + filename

        # S3에 이미지 업로드
        s3.upload_fileobj(image, S3_BUCKET, key)

        # 업로드된 이미지의 URL 반환
        image_url = f'https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}'
        return jsonify({'message': 'Success', 'image_url': image_url})
    except Exception as e:
        return jsonify({'message': 'Error', 'error': str(e)})
if __name__ == '__main__':  
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
