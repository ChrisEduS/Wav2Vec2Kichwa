import wandb

api = wandb.Api()
runs = api.runs(path="chrisedu19-universidad-san-francisco-de-quito/Wav2Vec2KichwaFineTuner")
for i in runs:
  print("run name = ",i.name," id: ", i.id)
