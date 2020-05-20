local dataset_path = "dataset/";

{
  "random_seed": 5,
  "numpy_seed": 5,
  "pytorch_seed": 5,
  "dataset_reader": {
    "type": "spider",
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "lazy": false,
    "cache_directory": "cache/train",
    "keep_if_unparsable": false,
    // "max_instances": null,
    "max_instances": 10,
  },
  "validation_dataset_reader": {
    "type": "spider",
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "cache_directory": "cache/val",
    "lazy": false,
    "keep_if_unparsable": true,
    // "max_instances": null,
    "max_instances": 10,
  },
  "train_data_path": dataset_path + "train_spider.json",
  "validation_data_path": dataset_path + "dev.json",
  "model": {
    "type": "spider",
    "world_encoder":{
      "type":"gnn",
        "encoder": {
          "type": "lstm",
          "input_size": 400,
          "hidden_size": 400,
          "bidirectional": true,
          "num_layers": 1
        },
        "entity_encoder": {
            "type": "boe",
            "embedding_dim": 200,
            "averaged": true
          },
        "question_embedder": {"token_embedders":{
              "tokens": {
                "type": "embedding",
                "embedding_dim": 200,
                "trainable": true
              }
        },

            },
      "action_embedding_dim": 200,
      "decoder_use_graph_entities": true,
      "gnn_timesteps": 3,
      "pruning_gnn_timesteps": 3,
      "parse_sql_on_decoding": true,
      "use_neighbor_similarity_for_linking": true,
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
    // "type": "basic",
    "batch_size" : 15
    // "batch_sampler" :{}
    
  },
  "validation_data_loader": {
    "batch_size" : 1,
  },
  "trainer": {
    "num_epochs": 100,
    "cuda_device": std.extVar('gpu'),
    "patience": 50,
    "validation_metric": "+sql_match",
    "optimizer": {
      "type": "adam",
      "lr": 0.001,
      "weight_decay": 5e-4
    },
    // "num_serialized_models_to_keep": 
    "checkpointer": {"num_serialized_models_to_keep": 2},
  }
}
