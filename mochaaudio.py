def pcm_to_wav(data: bytes, filename: str = 'output.wav') -> None:
	# http://soundfile.sapp.org/doc/WaveFormat/
	# compute data
	data_size = len(data)
	chunk_size = data_size + 36
	channels = 1
	sample_rate = 44100
	bits_per_sample = 32
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