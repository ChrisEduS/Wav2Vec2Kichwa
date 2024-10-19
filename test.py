import wandb 

run = wandb.init(project="Wav2Vec2KichwaFineTuner", job_type="upload")
# path to my remote data directory in Google Cloud Storage
test_data_sample = "/root/Wav2Vec2Kichwa/segmented_data/test/audio2_4_source2_p45_49_s57/audio2_4_source2_p45_49_s57.wav"
# create a regular artifact
dataset_at = wandb.Artifact('test_data_sample',type="raw_data")

audio_data = wandb.Audio(test_data_sample, sample_rate=48000, caption="audio2_4_source2_p45_49_s57")

# Log it to the run
run.log({"audio2_4_source2_p45_49_s57": audio_data})

# Finish the run
run.finish()