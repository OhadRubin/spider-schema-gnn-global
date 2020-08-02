local dataset_path = "dataset/";
local cache_path = "cache/new3";
local max_instances = null;
local gradient_acum = 4;
local batch_size = 6;
local max_steps = 90000;
local num_epochs = 125;

// local max_instances = 510;

{
  // "random_seed": 5,
  // "numpy_seed": 5,
  // "pytorch_seed": 5,
  "dataset_reader": {
    "type": "spider_ratsql",
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "lazy": false,
    "question_token_indexers":{
            "tokens":{
              "type":"pretrained_transformer",
                    "model_name":"bert-base-uncased",
            },

              },
    "cache_directory": cache_path + "train",
    "keep_if_unparsable": false,
    "max_instances": max_instances,
  },
  "validation_dataset_reader": {
    "type": "spider_ratsql",
    "question_token_indexers":{
            "tokens":{
              "type":"pretrained_transformer",
              "model_name":"bert-base-uncased",
            },
      },
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "cache_directory": cache_path + "val",
    "lazy": false,
    "keep_if_unparsable": true,
    "max_instances": max_instances,
  },
  "train_data_path": dataset_path + "train_spider.json",
  "validation_data_path": dataset_path + "dev.json",
  "model": {
    "type": "spider",
    "schema_encoder":{
      "type":"ratsql",
        // "encoder": {
        //   "type": "lstm",
        //   // "input_size": 1536,
        //   // "hidden_size": 1536,
        //   "input_size": 400,
        //   "hidden_size": 400,
          
        //   "bidirectional": true,
        //   "num_layers": 1
        // },
        // "entity_encoder": {
        //     "type": "boe",
        //     "embedding_dim": 200,
        //     // "embedding_dim": 768,
        //     "averaged": true
        //   },
        "question_embedder": {
        "token_embedders":{
          "tokens":{
                        "type":"pretrained_transformer",
                        "model_name":"bert-base-uncased",
          },

                          },
        



            },
      "action_embedding_dim": 300,
      // "action_embedding_dim": 768,
      // "decoder_use_graph_entities": true,
      // "gnn_timesteps": 3,
      // "pruning_gnn_timesteps": 3,
      "parse_sql_on_decoding": false,
      // "use_neighbor_similarity_for_linking": true,
      "dropout": 0.5,
    },
    "dataset_path": dataset_path,
    "decoder_beam_search": {
      "beam_size": 10
    },
    "training_beam_size": 1,
    "max_decoding_steps": 100,
    "input_attention": {"type": "dot_product"},
    "past_attention": {"type": "dot_product"},
    "dropout": 0.5
  },
  "data_loader": {
    "batch_size" : batch_size,
  },
  "validation_data_loader": {
    "batch_size" : 1,
  },
  "trainer": {
    "num_epochs": num_epochs,
    "cuda_device": std.extVar('gpu'),
    // "patience": 50,
    "validation_metric": "+sql_match",
    // "optimizer": {
    //   "type": "adam",
    //   "lr": 7.44e-4,
    //   // "lr": 0.0001,
    //   "parameter_groups": [
    //       [["question_embedder"], {"lr": 3e-6}]
    //       ],
    //   "weight_decay": 5e-4
    // },
        "optimizer": {
                      "type": "adam",
                      "lr": 7.44e-4,
                      // "lr": 0.0001,
                      "parameter_groups": [
                          [["question_embedder"], {"lr": 3e-6}]
                          ],
                      "weight_decay": 5e-4
                    },
  "learning_rate_scheduler":{

                    "type": "polynomial_decay",
                    "warmup_steps": std.floor(max_steps/20),
                    "power": 2.0,
  },

    "num_gradient_accumulation_steps" : 4,
    // "num_serialized_models_to_keep": 
    "checkpointer": {"num_serialized_models_to_keep": 2},
  }
}

