# AWS Config

### Jpyterの入り方
- ブラウザで、[http://ec2-52-91-223-39.compute-1.amazonaws.com:8888/](http://ec2-52-91-223-39.compute-1.amazonaws.com:8888/)を開く
- pass : okamotosaburo

### SSHでの接続方法
- `chmod 400 jupyter-test-1.pem`を実行して鍵ファイルのセキュリティを変更。
- `ssh -i "jupyter-test-1.pem" ec2-user@ec2-52-91-223-39.compute-1.amazonaws.com`を実行。
