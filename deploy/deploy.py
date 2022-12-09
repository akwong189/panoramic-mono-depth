import onnxruntime
import cv2
import numpy as np
import time
from jtop import jtop
import pickle

MAP_FILE = "./map_full_900.npy"
ONNX_MODEL = "mobile_diode_pano_ssim_l1_sobel_edges_old_preprocess.onnx"
STORE_FRAMES = True
STORE_STATS = True
GUI = False

def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=820,
    display_height=616,
    framerate=21,
    flip_method=2,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img / 255, 0).astype(np.float32)
    return img

def predict(img, sess, output_name, input_name):
    img = preprocess(img)
    try:
        res = sess.run([output_name], {input_name: img})
    except Exception as e:
        print(e)
        exit(1)
    res = np.squeeze(res)
    res = (res * 255).astype(np.uint8)
    res = np.expand_dims(res, -1)
    return res

def write_video(name, frames, frame_rate):
    print(frames[0].shape)
    height, width, channels = frames[0].shape
    if channels == 1:
        is_color = False
    else:
        is_color = True

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video = cv2.VideoWriter(name, fourcc, frame_rate, (width, height), is_color)

    for f in frames:
        video.write(f)
    print(f"Saved {name} to disk")
    video.release()

def stream(sess):
    i = 0
    xyd = np.load(MAP_FILE)
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    output = sess.get_outputs()[0].name
    inputs = sess.get_inputs()[0].name

    imgs = []
    depths = []
    fps = []
    stats = []


    assert cap.isOpened()

    start_time = None
    jetson = jtop()
    jetson.start()
    
    while True:
        ret, img = cap.read()
        start = time.time()
        img = cv2.remap(img, xyd, None, cv2.INTER_LINEAR)

        depth = predict(img, sess, output, inputs)
        if start_time is None:
            start_time = time.time()

        if GUI:
            #cv2.imshow("Panoramic Depth", depth)
            #keyCode = cv2.waitKey(1) & 0xFF
            # Stop the program on the ESC key
            #if keyCode == 27:
            #    break
            #if keyCode == 99:
            #    cv2.imwrite(f"./temp/{i}.png", img)
            #    i+=1
            #    print("Image Captured")
            pass

        fps.append(1 / (time.time() - start))
        if STORE_FRAMES:
            imgs.append(img)
            depths.append(depth)
        if STORE_STATS:
            stats.append(jetson.stats)
        if start_time is not None and time.time() - start_time > 30:
            break
        #print("fps: %0.2f" % (1 / (time.time() - start)))
    
    if STORE_FRAMES:
        avg_framerate = int(np.average(fps))
        write_video("frames.avi", imgs, avg_framerate)
        write_video("depths.avi", depths, avg_framerate)

        print(f"Average FPS: {np.average(fps)}")

    if STORE_STATS:
        file = open("run.history", "wb")
        pickle.dump({"fps": fps, "stats": stats}, file)

    jetson.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting test script")
    # Add onnx model here
    sess = onnxruntime.InferenceSession(ONNX_MODEL, providers=['TensorrtExecutionProvider'])
    print("Model Session Loaded")
    stream(sess)

