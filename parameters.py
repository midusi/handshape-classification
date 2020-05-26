thisdict_split ={
  "lsa16": 0.1,
  "Irish": 0.3,
  "rwth": 0.2,
  "Ciarp": 0.2,
  "indianA":0.15,
  "indianB":0.15,
  "jsl":0.2,
  "Nus1":0.1,
  "Nus2":0.15,
  "psl":0.1,
  "PugeaultASL_A":0.3,
  "PugeaultASL_B":0.3,
}
thisdict_batch_size_mobile =	{
  "lsa16": 32,
  "Irish": 64,
  "rwth": 64,
  "Ciarp": 64,
  "indianA":64,
  "indianB":32,
  "jsl": 64,
  "Nus1": 16,
  "Nus2": 32,
  "psl": 32,
  "PugeaultASL_A":64,
  "PugeaultASL_B":64,
}
thisdict_batch_size_dense =	{
  "lsa16": 32,
  "Irish": 32,
  "rwth": 32,
  "Ciarp": 32,
  "indianA":32,
  "indianB":32,
  "jsl":32,
  "Nus1":32,
  "Nus2":32,
  "psl":32,
  "PugeaultASL_A":32,
  "PugeaultASL_B":32,
}

thisdict_batch_size_eff =	{
  "lsa16": 32,
  "Irish": 64,
  "rwth": 64,
  "Ciarp": 64,
  "indianA":64,
  "indianB":32,
  "jsl": 64,
  "Nus1": 16,
  "Nus2": 32,
  "psl": 32,
  "PugeaultASL_A":64,
  "PugeaultASL_B":64,
}

def get_split_value(dataset_id):
    return thisdict_split[dataset_id]

def get_batch_mobile(dataset_id):
    return thisdict_batch_size_mobile[dataset_id]

def get_batch_dense(dataset_id):
    return thisdict_batch_size_dense[dataset_id]

def get_batch_eff(dataset_id):
  return thisdict_batch_size_eff[dataset_id]