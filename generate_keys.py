"""
- https://59travel.tistory.com/7

"""
import yaml
import streamlit_authenticator as stauth

usernames = ["kgh5408", "rkgus0524"]
names = ["김근형", "김가현"]
passwords = [] # yaml 파일 생성하고 비밀번호 지우기!
emails = ["kgh5408@nate.com", "rkgus0524@naver.com"]

hashed_passwords = [stauth.Hasher.hash(password) for password in passwords]

data = {
    "credentials" : {
        "usernames":{
            usernames[0]:{
                "name":names[0],
                "password":hashed_passwords[0],
                "email":emails[0]
                },
            usernames[1]:{
                "name":names[1],
                "password":hashed_passwords[1],
                "email":emails[1]
                },
            }
    },
    "cookie": {
        "expiry_days" : 0, # 만료일, 재인증 기능 필요없으면 0으로 세팅
        "key": "some_signature_key",
        "name" : "some_cookie_name"
    },
    "pre-authorized" : {
        "emails" : [
            "melsby@gmail.com"
        ]
    }
}

with open('config.yaml','w') as file:
    yaml.dump(data, file, default_flow_style=False)