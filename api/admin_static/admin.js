const state = {
  config: null,
  fields: new Map(),
  localStatus: new Map(),
  modelOptions: [],
  activeView: "providers",
};

const MASKED_SECRET = "********";
const VIEW_GROUPS = [
  {
    id: "providers",
    label: "Providers",
    title: "Providers",
    sections: ["providers", "runtime"],
    containerId: "providersSections",
  },
  {
    id: "model_config",
    label: "Model Config",
    title: "Model Config",
    sections: ["models", "thinking", "web_tools"],
    containerId: "modelConfigSections",
  },
  {
    id: "messaging",
    label: "Messaging",
    title: "Messaging",
    sections: ["messaging", "voice"],
    containerId: "messagingSections",
  },
  {
    id: "custom_providers",
    label: "Custom Providers",
    title: "Custom Providers",
    sections: [],
    containerId: "customProviderFormMount",
  },
];

const PROTOCOL_LABELS = {
  openai_chat: "OpenAI-compatible (Chat Completions)",
  anthropic_messages: "Anthropic (Messages API)",
};

const byId = (id) => document.getElementById(id);

function sourceLabel(source) {
  const labels = {
    default: "default",
    template: "template",
    repo_env: "repo .env",
    managed_env: "",
    explicit_env_file: "GDEC_ENV_FILE",
    process: "process env",
  };
  return Object.prototype.hasOwnProperty.call(labels, source) ? labels[source] : source;
}

function sourceText(field) {
  const parts = [];
  const label = sourceLabel(field.source);
  if (label) {
    parts.push(label);
  }
  if (field.locked) {
    parts.push("locked");
  }
  return parts.join(" ");
}

function providerName(providerId) {
  const names = {
    nvidia_nim: "NVIDIA NIM",
    open_router: "OpenRouter",
    deepseek: "DeepSeek",
    lmstudio: "LM Studio",
    llamacpp: "llama.cpp",
    ollama: "Ollama",
    kimi: "Kimi",
    wafer: "Wafer",
    opencode: "OpenCode Zen",
    zai: "Z.ai",
  };
  if (names[providerId]) return names[providerId];
  return providerId
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function statusClass(status) {
  if (["configured", "reachable", "running"].includes(status)) return "ok";
  if (["missing_key", "missing_url", "unknown"].includes(status)) return "warn";
  if (["offline", "error"].includes(status)) return "error";
  return "neutral";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function load() {
  showMessage("Loading admin config");
  const config = await api("/admin/api/config");
  state.config = config;
  state.fields = new Map(config.fields.map((field) => [field.key, field]));
  renderNav();
  renderProviders(config.provider_status);
  renderSections(config.sections, config.fields);
  await loadCustomProviders();
  byId("configPath").textContent = config.paths.managed;
  await validate(false);
  await refreshLocalStatus();
  updateDirtyState();
  showMessage("");
}

function renderNav() {
  const nav = byId("sectionNav");
  nav.innerHTML = "";
  VIEW_GROUPS.forEach((view, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `nav-link${index === 0 ? " active" : ""}`;
    button.dataset.view = view.id;
    button.textContent = view.label;
    if (index === 0) {
      button.setAttribute("aria-current", "page");
    }
    button.addEventListener("click", () => {
      setActiveView(view.id, { scroll: true });
    });
    nav.appendChild(button);
  });
  setActiveView(state.activeView, { scroll: false });
}

function setActiveView(viewId, { scroll = false } = {}) {
  const activeView =
    VIEW_GROUPS.find((view) => view.id === viewId) || VIEW_GROUPS[0];
  state.activeView = activeView.id;
  byId("pageTitle").textContent = activeView.title;

  document.querySelectorAll(".nav-link").forEach((link) => {
    const selected = link.dataset.view === activeView.id;
    link.classList.toggle("active", selected);
    if (selected) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });

  document.querySelectorAll(".admin-view").forEach((view) => {
    const selected = view.dataset.view === activeView.id;
    view.classList.toggle("active", selected);
    view.hidden = !selected;
  });

  if (scroll) {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

function renderProviders(providerStatus) {
  const grid = byId("providerGrid");
  grid.innerHTML = "";
  providerStatus.forEach((provider) => {
    const card = document.createElement("article");
    card.className = "provider-card";
    card.dataset.provider = provider.provider_id;

    const title = document.createElement("div");
    title.className = "provider-title";
    title.innerHTML = `<strong>${providerName(provider.provider_id)}</strong>`;

    const pill = document.createElement("span");
    pill.className = `status-pill ${statusClass(provider.status)}`;
    pill.textContent = provider.label;
    title.appendChild(pill);

    const meta = document.createElement("div");
    meta.className = "provider-meta";
    meta.textContent =
      provider.kind === "local"
        ? provider.base_url || "No local URL configured"
        : provider.credential_env;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "test-button";
    button.textContent = provider.kind === "local" ? "Test" : "Refresh models";
    button.addEventListener("click", () => testProvider(provider.provider_id, button));

    card.append(title, meta, button);
    grid.appendChild(card);
  });
}

function updateProviderCard(providerId, status, label, metaText) {
  const card = document.querySelector(`[data-provider="${providerId}"]`);
  if (!card) return;
  const pill = card.querySelector(".status-pill");
  pill.className = `status-pill ${statusClass(status)}`;
  pill.textContent = label;
  if (metaText) {
    card.querySelector(".provider-meta").textContent = metaText;
  }
}

function renderSections(sections, fields) {
  VIEW_GROUPS.forEach((view) => {
    byId(view.containerId).innerHTML = "";
  });

  const sectionById = new Map(sections.map((section) => [section.id, section]));
  const bySection = new Map();
  sections.forEach((section) => bySection.set(section.id, []));
  fields.forEach((field) => {
    if (!bySection.has(field.section)) bySection.set(field.section, []);
    bySection.get(field.section).push(field);
  });

  VIEW_GROUPS.forEach((view) => {
    const container = byId(view.containerId);
    view.sections.forEach((sectionId) => {
      const section = sectionById.get(sectionId);
      const sectionFields = bySection.get(sectionId) || [];
      if (!section || sectionFields.length === 0) return;

      const sectionEl = document.createElement("section");
      sectionEl.className = "settings-section";
      sectionEl.id = `section-${section.id}`;

      const heading = document.createElement("div");
      heading.className = "section-heading";
      heading.innerHTML = `<div><h3>${section.label}</h3><p>${section.description}</p></div>`;
      sectionEl.appendChild(heading);

      const grid = document.createElement("div");
      grid.className = "field-grid";
      sectionFields.forEach((field) => {
        grid.appendChild(renderField(field));
      });
      sectionEl.appendChild(grid);

      if (sectionFields.some((field) => field.advanced)) {
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "ghost-button advanced-toggle";
        toggle.textContent = "Show advanced";
        toggle.addEventListener("click", () => {
          const showing = sectionEl.classList.toggle("show-advanced");
          toggle.textContent = showing ? "Hide advanced" : "Show advanced";
        });
        sectionEl.appendChild(toggle);
      }

      container.appendChild(sectionEl);
    });
  });
}

function renderField(field) {
  const wrapper = document.createElement("div");
  wrapper.className = `field${field.advanced ? " advanced-field" : ""}`;
  wrapper.dataset.key = field.key;

  const label = document.createElement("label");
  label.htmlFor = `field-${field.key}`;
  const labelText = document.createElement("span");
  labelText.textContent = field.label;
  label.appendChild(labelText);

  const source = sourceText(field);
  if (source) {
    const sourceEl = document.createElement("span");
    sourceEl.className = "field-source";
    sourceEl.textContent = source;
    label.appendChild(sourceEl);
  }

  const input = inputForField(field);
  input.id = `field-${field.key}`;
  input.dataset.key = field.key;
  input.dataset.original = field.value || "";
  input.dataset.secret = field.secret ? "true" : "false";
  input.dataset.configured = field.configured ? "true" : "false";
  input.disabled = field.locked;
  input.addEventListener("input", updateDirtyState);
  input.addEventListener("change", updateDirtyState);

  wrapper.append(label, input);
  if (field.description) {
    const description = document.createElement("div");
    description.className = "field-description";
    description.textContent = field.description;
    wrapper.appendChild(description);
  }
  return wrapper;
}

function inputForField(field) {
  if (field.type === "boolean") {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = String(field.value).toLowerCase() === "true";
    input.dataset.original = input.checked ? "true" : "false";
    return input;
  }

  if (field.type === "tri_boolean") {
    const select = document.createElement("select");
    [
      ["", "Inherit"],
      ["true", "Enabled"],
      ["false", "Disabled"],
    ].forEach(([value, label]) => select.appendChild(option(value, label)));
    select.value = field.value || "";
    return select;
  }

  if (field.type === "select") {
    const select = document.createElement("select");
    field.options.forEach((value) => select.appendChild(option(value, value)));
    select.value = field.value || field.options[0] || "";
    return select;
  }

  if (field.type === "textarea") {
    const textarea = document.createElement("textarea");
    textarea.value = field.value || "";
    return textarea;
  }

  const input = document.createElement("input");
  input.type = field.type === "number" ? "number" : "text";
  if (field.type === "secret") {
    input.type = "password";
    input.placeholder = field.configured
      ? "Configured - enter a new value to replace"
      : "Not configured";
    input.value = "";
    input.autocomplete = "off";
  } else {
    input.value = field.value || "";
  }
  if (field.key.startsWith("MODEL")) {
    input.setAttribute("list", "model-options");
  }
  return input;
}

function option(value, label) {
  const optionEl = document.createElement("option");
  optionEl.value = value;
  optionEl.textContent = label;
  return optionEl;
}

function readFieldValue(input) {
  if (input.type === "checkbox") return input.checked ? "true" : "false";
  if (input.dataset.secret === "true" && input.dataset.configured === "true") {
    return input.value ? input.value : MASKED_SECRET;
  }
  return input.value;
}

function changedValues() {
  const values = {};
  document.querySelectorAll("[data-key]").forEach((input) => {
    if (input.disabled || !input.matches("input, select, textarea")) return;
    const value = readFieldValue(input);
    if (value !== input.dataset.original) {
      values[input.dataset.key] = value;
    }
  });
  return values;
}

function updateDirtyState() {
  const count = Object.keys(changedValues()).length;
  byId("dirtyState").textContent =
    count === 0 ? "No changes" : `${count} unsaved change${count === 1 ? "" : "s"}`;
  byId("applyButton").disabled = count === 0;
}

async function validate(showResult = true) {
  const result = await api("/admin/api/config/validate", {
    method: "POST",
    body: JSON.stringify({ values: changedValues() }),
  });
  if (showResult) {
    showValidationResult(result);
  }
  return result;
}

function showValidationResult(result) {
  if (result.valid) {
    showMessage("Config shape is valid", "ok");
  } else {
    showMessage(result.errors.join("; "), "error");
  }
}

async function apply() {
  const result = await api("/admin/api/config/apply", {
    method: "POST",
    body: JSON.stringify({ values: changedValues() }),
  });
  if (!result.applied) {
    showValidationResult(result);
    return;
  }
  const restart = result.restart || {};
  if (restart.required && restart.automatic) {
    showMessage("Applied. Restarting server...", "ok");
    byId("applyButton").disabled = true;
    setTimeout(() => {
      window.location.href = restart.admin_url || "/admin";
    }, 1600);
    return;
  }
  const pending = restart.required ? restart.fields || [] : result.pending_fields || [];
  await load();
  showMessage(
    pending.length
      ? `Applied. Restart gdec-server to use: ${pending.join(", ")}`
      : "Applied",
    "ok",
  );
}

async function refreshLocalStatus() {
  const result = await api("/admin/api/providers/local-status");
  result.providers.forEach((provider) => {
    state.localStatus.set(provider.provider_id, provider);
    const meta = provider.status_code
      ? `${provider.base_url} returned HTTP ${provider.status_code}`
      : provider.base_url;
    updateProviderCard(provider.provider_id, provider.status, provider.label, meta);
  });
}

async function testProvider(providerId, button) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Testing";
  try {
    const result = await api(`/admin/api/providers/${providerId}/test`, {
      method: "POST",
      body: "{}",
    });
    if (result.ok) {
      updateProviderCard(
        providerId,
        "reachable",
        `${result.models.length} models`,
        result.models.slice(0, 3).join(", ") || "No models returned",
      );
      state.modelOptions = Array.from(
        new Set([
          ...state.modelOptions,
          ...result.models.map((model) => `${providerId}/${model}`),
        ]),
      ).sort();
      syncModelDatalist();
    } else {
      updateProviderCard(providerId, "offline", result.error_type, result.error_type);
    }
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function syncModelDatalist() {
  let datalist = byId("model-options");
  if (!datalist) {
    datalist = document.createElement("datalist");
    datalist.id = "model-options";
    document.body.appendChild(datalist);
  }
  datalist.innerHTML = "";
  state.modelOptions.forEach((model) => datalist.appendChild(option(model, model)));
}

function showMessage(message, kind = "") {
  const area = byId("messageArea");
  area.textContent = message;
  area.className = `message-area ${kind}`.trim();
}

async function loadCustomProviders() {
  let data;
  try {
    data = await api("/admin/api/custom-providers");
  } catch (error) {
    showMessage(error.message, "error");
    return;
  }
  renderCustomProviders(data.providers || []);
}

function renderCustomProviders(providers) {
  const list = byId("customProviderList");
  list.innerHTML = "";
  if (providers.length === 0) {
    const empty = document.createElement("p");
    empty.className = "provider-meta";
    empty.textContent = "No custom providers yet.";
    list.appendChild(empty);
    return;
  }
  providers.forEach((provider) => {
    const card = document.createElement("article");
    card.className = "provider-card";

    const title = document.createElement("div");
    title.className = "provider-title";
    title.innerHTML = `<strong></strong>`;
    title.querySelector("strong").textContent = provider.display_name;
    const pill = document.createElement("span");
    pill.className = "status-pill neutral";
    pill.textContent = provider.provider_id;
    title.appendChild(pill);

    const meta = document.createElement("div");
    meta.className = "provider-meta";
    const protocol = PROTOCOL_LABELS[provider.protocol] || provider.protocol;
    const keyState = provider.has_api_key ? "key set" : "no key";
    const modelCount = (provider.models || []).length;
    const modelsPart =
      modelCount > 0
        ? ` · ${modelCount} model${modelCount === 1 ? "" : "s"}`
        : "";
    meta.textContent = `${protocol} · ${provider.base_url} · ${keyState}${modelsPart}`;

    const actions = document.createElement("div");
    actions.className = "cp-actions";
    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "test-button";
    edit.textContent = "Edit";
    edit.addEventListener("click", () => showCustomProviderForm(provider));
    const del = document.createElement("button");
    del.type = "button";
    del.className = "test-button";
    del.textContent = "Delete";
    del.addEventListener("click", () =>
      deleteCustomProvider(provider.provider_id),
    );
    actions.append(edit, del);

    card.append(title, meta, actions);
    list.appendChild(card);
  });
}

function _customProviderField(labelText, input) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  const label = document.createElement("label");
  const span = document.createElement("span");
  span.textContent = labelText;
  label.appendChild(span);
  wrapper.append(label, input);
  return wrapper;
}

function showCustomProviderForm(record) {
  const editing = Boolean(record);
  const mount = byId("customProviderFormMount");
  mount.innerHTML = "";

  const form = document.createElement("form");
  form.className = "settings-section";

  const heading = document.createElement("div");
  heading.className = "section-heading";
  const headingInner = document.createElement("div");
  const headingTitle = document.createElement("h3");
  headingTitle.textContent = `${editing ? "Edit" : "Add"} custom provider`;
  const headingHint = document.createElement("p");
  headingHint.textContent =
    "Route models with provider_id/model. Provider ID is permanent.";
  headingInner.append(headingTitle, headingHint);
  heading.appendChild(headingInner);
  form.appendChild(heading);

  const grid = document.createElement("div");
  grid.className = "field-grid";

  const idInput = document.createElement("input");
  idInput.type = "text";
  idInput.autocomplete = "off";
  idInput.placeholder = "my-llm";
  idInput.value = record ? record.provider_id : "";
  idInput.disabled = editing;

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.autocomplete = "off";
  nameInput.placeholder = "My LLM";
  nameInput.value = record ? record.display_name : "";

  const urlInput = document.createElement("input");
  urlInput.type = "text";
  urlInput.autocomplete = "off";
  urlInput.placeholder = "https://api.example.com/v1";
  urlInput.value = record ? record.base_url : "";

  const keyInput = document.createElement("input");
  keyInput.type = "password";
  keyInput.autocomplete = "off";
  keyInput.placeholder = "sk-... (optional)";
  keyInput.value = record && record.has_api_key ? MASKED_SECRET : "";

  const protocolInput = document.createElement("select");
  [
    ["openai_chat", PROTOCOL_LABELS.openai_chat],
    ["anthropic_messages", PROTOCOL_LABELS.anthropic_messages],
  ].forEach(([value, text]) => {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = text;
    protocolInput.appendChild(opt);
  });
  protocolInput.value = record ? record.protocol : "openai_chat";

  const bypassProxyInput = document.createElement("input");
  bypassProxyInput.type = "checkbox";
  bypassProxyInput.checked = Boolean(record && record.bypass_system_proxy);

  const verifyTlsInput = document.createElement("input");
  verifyTlsInput.type = "checkbox";
  verifyTlsInput.checked = record ? record.verify_tls !== false : true;

  grid.append(
    _customProviderField("Provider ID", idInput),
    _customProviderField("Display Name", nameInput),
    _customProviderField("Default Base URL", urlInput),
    _customProviderField("API Key (optional)", keyInput),
    _customProviderField("Protocol", protocolInput),
    _customProviderField("Bypass system proxy", bypassProxyInput),
    _customProviderField("Verify TLS (uncheck for lab self-signed)", verifyTlsInput),
  );
  form.appendChild(grid);

  const modelsSection = document.createElement("section");
  modelsSection.className = "settings-section";
  const modelsHeading = document.createElement("div");
  modelsHeading.className = "section-heading";
  const modelsHeadingInner = document.createElement("div");
  const modelsTitle = document.createElement("h3");
  modelsTitle.textContent = "Models";
  const modelsHint = document.createElement("p");
  modelsHint.textContent =
    "Manual model list. Required when the provider does not expose /models.";
  modelsHeadingInner.append(modelsTitle, modelsHint);
  modelsHeading.appendChild(modelsHeadingInner);
  modelsSection.appendChild(modelsHeading);

  const modelsList = document.createElement("div");
  modelsList.className = "cp-models";
  modelsSection.appendChild(modelsList);

  function appendModelRow(model) {
    const row = document.createElement("div");
    row.className = "cp-model-row";

    const modelIdInput = document.createElement("input");
    modelIdInput.type = "text";
    modelIdInput.placeholder = "model id (e.g. gpt-4o)";
    modelIdInput.className = "cp-model-id";
    modelIdInput.value = model ? model.model_id : "";

    const modelNameInput = document.createElement("input");
    modelNameInput.type = "text";
    modelNameInput.placeholder = "display name (e.g. GPT-4o)";
    modelNameInput.className = "cp-model-name";
    modelNameInput.value = model ? model.display_name : "";

    const status = document.createElement("span");
    status.className = "cp-model-status";

    const testBtn = document.createElement("button");
    testBtn.type = "button";
    testBtn.className = "test-button";
    testBtn.textContent = "Test";
    testBtn.disabled = !editing;
    testBtn.title = editing
      ? "Send a one-token request to verify this model"
      : "Save provider first, then test";
    testBtn.addEventListener("click", async () => {
      const modelId = modelIdInput.value.trim();
      if (!modelId) {
        status.textContent = "model id required";
        status.className = "cp-model-status error";
        return;
      }
      status.textContent = "Testing…";
      status.className = "cp-model-status";
      let result;
      try {
        result = await api(
          `/admin/api/custom-providers/${encodeURIComponent(record.provider_id)}/models/test`,
          {
            method: "POST",
            body: JSON.stringify({ model_id: modelId }),
          },
        );
      } catch (err) {
        status.textContent = err.message;
        status.className = "cp-model-status error";
        return;
      }
      if (result.ok) {
        status.textContent = `OK (${result.model || modelId})`;
        status.className = "cp-model-status ok";
      } else {
        const summary = result.message || result.error_type || "failed";
        status.textContent = `${result.error_type || "Error"}: ${summary}`;
        status.className = "cp-model-status error";
      }
    });

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "test-button";
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", () => {
      row.remove();
    });

    row.append(modelIdInput, modelNameInput, status, testBtn, removeBtn);
    modelsList.appendChild(row);
  }

  (record && record.models ? record.models : []).forEach(appendModelRow);

  const addModelBtn = document.createElement("button");
  addModelBtn.type = "button";
  addModelBtn.className = "secondary-button";
  addModelBtn.textContent = "+ Add model";
  addModelBtn.addEventListener("click", () => appendModelRow(null));
  modelsSection.appendChild(addModelBtn);
  form.appendChild(modelsSection);

  const actions = document.createElement("div");
  actions.className = "cp-actions";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "secondary-button";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => {
    mount.innerHTML = "";
  });
  const save = document.createElement("button");
  save.type = "submit";
  save.className = "primary-button";
  save.textContent = "Save";
  actions.append(cancel, save);

  const error = document.createElement("div");
  error.className = "message-area";

  form.append(actions, error);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    error.textContent = "";
    error.className = "message-area";
    const models = Array.from(modelsList.querySelectorAll(".cp-model-row"))
      .map((row) => ({
        model_id: row.querySelector(".cp-model-id").value.trim(),
        display_name: row.querySelector(".cp-model-name").value.trim(),
      }))
      .filter((m) => m.model_id || m.display_name);
    const payload = {
      provider_id: idInput.value.trim(),
      display_name: nameInput.value.trim(),
      base_url: urlInput.value.trim(),
      api_key: keyInput.value,
      protocol: protocolInput.value,
      models,
      bypass_system_proxy: bypassProxyInput.checked,
      verify_tls: verifyTlsInput.checked,
    };
    let result;
    try {
      result = await api("/admin/api/custom-providers", {
        method: "POST",
        body: JSON.stringify(payload),
      });
    } catch (err) {
      error.textContent = err.message;
      error.className = "message-area error";
      return;
    }
    if (!result.applied) {
      error.textContent = (result.errors || ["Save failed"]).join("; ");
      error.className = "message-area error";
      return;
    }
    mount.innerHTML = "";
    showMessage("Custom provider saved", "ok");
    await load();
  });

  mount.appendChild(form);
  mount.scrollIntoView({ behavior: "smooth" });
}

async function deleteCustomProvider(providerId) {
  let result;
  try {
    result = await api(
      `/admin/api/custom-providers/${encodeURIComponent(providerId)}`,
      { method: "DELETE" },
    );
  } catch (error) {
    showMessage(error.message, "error");
    return;
  }
  if (!result.applied) {
    showMessage((result.errors || ["Delete failed"]).join("; "), "error");
    return;
  }
  showMessage(`Deleted ${providerId}`, "ok");
  await load();
}

byId("customProviderAdd").addEventListener("click", () =>
  showCustomProviderForm(null),
);
byId("validateButton").addEventListener("click", () => validate(true));
byId("applyButton").addEventListener("click", apply);

load().catch((error) => {
  showMessage(error.message, "error");
});
