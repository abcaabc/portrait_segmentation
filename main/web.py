from flask import Flask, render_template, request, redirect, url_for,  Response, flash
from segmentation import segment

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the key is unique and secret'
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'GET':
        return render_template('upload.html')
    elif request.method =='POST':
        file = request.files['image']
        if file.filename and 'image' == file.content_type.split('/')[0]:
            img_path = f'upload_images/{file.filename}'
            file.save(img_path)
            success, buffer = segment(img_path)
            
            if not success:
                flash('图片编码失败')
                return redirect(url_for('upload_image'))
            img_bytes = buffer.tobytes()
            response = Response(img_bytes, mimetype='image/png')
            return response
        else:
            flash('文件未选择')
            return redirect(url_for('upload_image'))


if __name__ == '__main__':
    app.run(debug=True,port=5000,host='0.0.0.0')
