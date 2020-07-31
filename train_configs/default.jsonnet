local dataset_path = "dataset/";
local cache_path = "cache/new5";
local max_instances = null;
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
      "parse_sql_on_decoding": true,
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
    "batch_size" : 10,
  },
  "validation_data_loader": {
    "batch_size" : 1,
  },
  "trainer": {
    "num_epochs": 1000,
    "cuda_device": std.extVar('gpu'),
    // "patience": 50,
    "validation_metric": "+sql_match",
    "optimizer": {
      "type": "adam",
      "lr": 0.001,
      // "lr": 0.0001,
      "weight_decay": 5e-4
    },
    // "num_serialized_models_to_keep": 
    "checkpointer": {"num_serialized_models_to_keep": 2},
  }
}
