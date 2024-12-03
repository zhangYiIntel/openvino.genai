// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <filesystem>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov::genai {

/// @brief A map of models for VLMPipeline constructor. 
/// Key is model name (e.g. "vision_embeddings", "text_embeddings", "language", "resampler")
/// and value is a pair of model IR as string and weights as tensor.
using ModelsMap = std::map<std::string, std::pair<std::string, ov::Tensor>>;

/// @brief A Visual language modeling pipeline class used to generate a
/// response or run a chat given a prompt and an image.
class OPENVINO_GENAI_EXPORTS VLMPipeline {
public:
    /// @brief Construct a pipeline from a folder containing tokenizer
    /// and model IRs.
    /// @param models_path A folder to read tokenizer and model IRs.
    /// @param device Inference device. A tokenizer is always compiled
    /// for CPU.
    /// @param properties A config to pass to ov::Core::compile_model().
    VLMPipeline(
        const std::filesystem::path& models_path,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    /// @brief Construct a pipeline from a map of models and their weights.
    /// @param models_map A map where key is model name (e.g. "vision_embeddings", "text_embeddings", "language", "resampler")
    /// and value is a pair of model IR as string and weights as tensor.
    /// @param tokenizer A tokenizer.
    /// @param config_dir_path A path to directory containing config.json.
    /// @param device Inference device. A tokenizer is always compiled
    /// for CPU.
    /// @param properties A config to pass to ov::Core::compile_model().
    /// @param generation_config Optional generation configuration for the pipeline.
    VLMPipeline(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties = {},
        const ov::genai::GenerationConfig& generation_config = {}
    );

    /// @brief Construct a pipeline from a folder containing tokenizer
    /// and model IRs. Accepts arbitrary list of optional properties.
    /// @param models_path A folder to read tokenizer and model IRs.
    /// @param device Inference device. A tokenizer is always compiled
    /// for CPU.
    /// @param properties A config to pass to ov::Core::compile_model().
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    VLMPipeline(
        const std::filesystem::path& models_path,
        const std::string& device,
        Properties&&... properties)
        : VLMPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    /// @brief Construct a pipeline from a map of models and their weights.
    /// @param models_map A map where key is model name (e.g. "vision_embeddings", "text_embeddings", "language", "resampler")
    /// and value is a pair of model IR as string and weights as tensor.
    /// @param tokenizer A tokenizer.
    /// @param config_dir_path A path to directory containing config.json.
    /// @param device Inference device. A tokenizer is always compiled
    /// for CPU.
    /// @param properties A config to pass to ov::Core::compile_model().
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    VLMPipeline(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        Properties&&... properties)
        : VLMPipeline(models_map, tokenizer, config_dir_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    /// @brief Default destructor.
    ~VLMPipeline();

    /// @brief Generate a response given a prompt and any number of
    /// uint8 RGB images with [NHWC] or [HWC] layout.
    /// @param prompt A prompt to respond to.
    /// @param images Images to be prepended to a prompt.
    /// @param generation_config A config to follow for text generation.
    /// @param streamer A streamer to acquire intermediate result.
    /// @return A string generated by a model.
    DecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& rgbs,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    );

    /// @brief Generate a response given a prompt and config.
    /// @param prompt A prompt to respond to.
    /// @param config_map A config may contain GenerationConfig, values
    /// for its members, StreamerVariant a single image or multiple
    /// images.
    /// @return A string generated by a model.
    DecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    );

    /// @brief Generate a response given a prompt and arbitrary number
    /// of ov::Property instances.
    /// Example:
    /// generate("text", image(rgb), do_sample(true));
    /// @param prompt A prompt to respond to.
    /// @param ...properties ov::Property instances to be combined into
    /// ov::AnyMap.
    /// @return A string generated by a model.
    template <typename... Properties>
    util::EnableIfAllStringAny<DecodedResults, Properties...> generate(
        const std::string& prompt,
        Properties&&... properties
    ) {
        return generate(
            prompt, AnyMap{std::forward<Properties>(properties)...}
        );
    }

    /// @brief Activate chat mode. Chat preserves previous history and
    /// applies chat_template to input prompts. Calling start_chat()
    /// again or finish_chat() drops the memorized history.
    /// It's possible to disable
    /// chat_template application by calling
    /// set_chat_template("{% for message in messages %}{{ message['content'] }}{% endfor %}")
    /// @param system_message Some chat_templates contain system role
    /// in addition to user and assistant roles. Set a message for that
    /// role.
    void start_chat(const std::string& system_message="");

    /// @brief Deactivate chat mode.
    void finish_chat();

    /// @brief Set a custom chat template. Can be used to deactivate
    /// chat_template application for chat mode if called with
    /// "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    /// or workaround unsupported chat_template entries in a default
    /// model chat_template.
    /// @param new_template A new template to override with.
    void set_chat_template(const std::string& new_template);

    /// @brief Get a Tokenizer used to tokenize input and detokenize
    /// output.
    ov::genai::Tokenizer get_tokenizer() const;

    /// @brief Extract GenerationConfig used to get default values.
    /// @return Default values used.
    GenerationConfig get_generation_config() const;

    /// @brief Override default values for GenerationConfig
    /// @param new_config A config to override default values with.
    void set_generation_config(const GenerationConfig& new_config);

private:
    class VLMPipelineImpl;
    std::unique_ptr<VLMPipelineImpl> m_pimpl;
};

/*
 * utils that allow to use generate() in the following way:
 * pipe.generate(prompt, ov::genai::image(image_tensor)).
*/
static constexpr ov::Property<ov::Tensor> image{"image"};
static constexpr ov::Property<std::vector<ov::Tensor>> images{"images"};
}
