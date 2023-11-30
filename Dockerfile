# Python 이미지를 사용합니다.
FROM python:3.10.11

# 작업 디렉터리를 설정합니다.
WORKDIR /usr/src/app

# requirements.txt 파일을 컨테이너로 복사합니다.
COPY requirements.txt ./

# 필요한 Python 패키지들을 설치합니다.
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx



# 애플리케이션 코드를 컨테이너로 복사합니다.
COPY . .

# Flask 서버의 기본 포트를 설정합니다.
EXPOSE 5000

# 애플리케이션을 실행합니다.
CMD [ "python", "./aiEditKakao.py" ]
