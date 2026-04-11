import streamlit_authenticator as stauth

passwords = ["demo123", "admin123", "judge123"]

hashed = [stauth.Hasher().hash(pwd) for pwd in passwords]

for h in hashed:
    print(h)