def pcm_to_wav(data: bytes, filename: str = 'output.wav', bits_per_sample: int = 16, \
		sample_rate: int = 44100) -> None:
	# http://soundfile.sapp.org/doc/WaveFormat/
	# compute data
	data_size = len(data)
	chunk_size = data_size + 36
	channels = 1
	# construct header
	header = b'RIFF' + \
		chunk_size.to_bytes(4, byteorder='little') + \
		b'WAVEfmt ' + \
		(16).to_bytes(4, byteorder='little') + \
		(1).to_bytes(2, byteorder='little') + \
		(channels).to_bytes(2, byteorder='little') + \
		(sample_rate).to_bytes(4, byteorder='little') + \
		(sample_rate * channels * bits_per_sample//8).to_bytes(4, byteorder='little') + \
		(channels * bits_per_sample//8).to_bytes(2, byteorder='little') + \
		(bits_per_sample).to_bytes(2, byteorder='little') + \
		b'data' + \
		(data_size).to_bytes(4, byteorder='little')
	# write to file
	with open(filename, 'wb+') as file:
		file.write(header + data)

def play_file(filename: str) -> int:
	"""Outputs console error code, if any, as int"""
	from os import name as os_name
	if os_name == 'nt': # Windows NT
		from winsound import PlaySound, SND_FILENAME
		PlaySound(filename, SND_FILENAME)
		return 0
	if os_name == 'posix': # Linux OR Mac... probably
		from os import system
		from platform import system as get_os_name
		if get_os_name() == 'Linux':
			return system(f'aplay {filename}')
		if get_os_name() == 'Darwin': # Mac
			return system(f'afplay {filename}')
		if get_os_name() == 'Haiku':
			return system(f'media_client play {filename}')
		if get_os_name()[-3:] == 'BSD':
			for command in 'vlc aplay cmus moc mpv mplayer mplayer2'.split():
				if not system(f'{command} {filename}'):
					return 0
			else:
				return 32512
	elif os_name == 'os2':
		return system(f'fmgplay {filename}')
	# unknown OS, try to use pygame
	import pygame # unfortunately no 32-bit support
	# play audio file with pygame
	pygame.mixer.init()
	pygame.mixer.music.load(filename)
	pygame.mixer.music.play()
	pygame.quit()
	return 0