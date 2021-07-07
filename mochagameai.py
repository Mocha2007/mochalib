import pynput # https://pynput.readthedocs.io/en/latest/keyboard.html#monitoring-the-keyboard
from mss import mss # https://python-mss.readthedocs.io/examples.html#pil ; https://pypi.org/project/mss/
from PIL import Image #, ImageGrab

# todo: AIs to play various flash games

def maybe_terminate(key: pynput.keyboard.Key) -> None:
	if key != pynput.keyboard.Key.esc:
		return
	exit() # terminate program IMMEDIATELY

# listener = pynput.keyboard.Listener(on_press=maybe_terminate)
# listener.start()

def screenshot() -> Image.Image:
	with mss() as sct:
		sct.shot(output='mgai_debug/mgai-{date}') # saves to mgai-(date).png
		raw_pixels = sct.grab(sct.monitors[1]) # can also do (left, top, width, height) as an arg
		img = Image.frombytes("RGB", raw_pixels.size, raw_pixels.bgra, "raw", "BGRX")
	return img

def test() -> None:
	# im = Image.open('mgai.png').load()
	im = screenshot()
	x = y = 0
	im[x, y] # gets pixel at x, y as (r, g, b)
	# ImageGrab.grab() # get screenshot
	# bbox=(left, top, right, bottom)

# https://www.geeksforgeeks.org/with-statement-in-python/
# can create objects that can utilize with. might be useful for opening/closing menus in some games?