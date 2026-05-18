import {
  ArrowUp,
  BarChart3,
  CreditCard,
  Menu,
  Package,
  PanelLeftClose,
  Plus,
  RefreshCw,
  Search,
  Square,
  TrendingUp,
  Trash2,
  Truck
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

const navItems = [
  {
    label: "Products",
    icon: Package,
    question: "top sản phẩm năm 2018"
  },
  {
    label: "Delivery",
    icon: Truck,
    question: "giao hàng trễ ảnh hưởng thế nào đến đánh giá của khách hàng năm 2018?"
  }
];

const quickActions = [
  {
    label: "Phân tích danh mục",
    icon: BarChart3,
    question: "phân tích 5 danh mục sản phẩm tốt nhất năm 2018 dựa trên doanh thu, số đơn và review"
  },
  {
    label: "So sánh doanh thu",
    icon: TrendingUp,
    question: "doanh thu quý 1, quý 2, quý 3 năm 2018 khác nhau thế nào?"
  },
  {
    label: "Tra cứu đơn hàng",
    icon: CreditCard,
    question: "đơn e481f51cbdc54678b7cc49136f2d6af7 được thanh toán bằng phương thức nào?"
  }
];

function createId() {
  return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createConversation(title = "New chat", messages = []) {
  const now = Date.now();
  return {
    id: createId(),
    title,
    messages,
    createdAt: now,
    updatedAt: now
  };
}

function titleFromQuestion(question) {
  const clean = question.trim().replace(/\s+/g, " ");
  return clean.length > 54 ? `${clean.slice(0, 54)}...` : clean || "New chat";
}

function timeValue(value) {
  if (typeof value === "number") return value;
  const parsed = Date.parse(value || "");
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatNumber(value) {
  if (typeof value !== "number") return value ?? "";
  return new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 2 }).format(value);
}

function looksLikeBadAnalysis(text) {
  return /(?:\*\*)?\s*(?:cảnh|canh|kịch\s*bản|kich\s*ban)\s*\d+|câu\s*hỏi\s*phân\s*tích\s*dữ\s*liệu|cau\s*hoi\s*phan\s*tich\s*du\s*lieu|phân\s*tích\s*5\s*danh\s*mục\s*sản\s*phẩm\s*tốt\s*nhất|phan\s*tich\s*5\s*danh\s*muc\s*san\s*pham\s*tot\s*nhat/i.test(String(text || ""));
}

function answerText(result) {
  if (!result) return "";
  if (result.needs_clarification) {
    return result.clarifying_question || result.reason || "Cần bổ sung thông tin.";
  }
  if (result.analysis && !(result.safe_summary && looksLikeBadAnalysis(result.analysis))) return result.analysis;
  if (result.answer) return result.answer;
  if (result.safe_summary) return result.safe_summary;
  if (result.error) return result.error;
  if (result.text) return result.text;
  if (result.ok) return "Đã nhận được kết quả từ agent.";
  return "Không có dữ liệu trả lời.";
}

function resultTables(result) {
  if (!result || typeof result !== "object") return [];
  const tables = [];
  const add = (title, rows) => {
    if (Array.isArray(rows) && rows.length) tables.push({ title, rows });
  };

  add("Kết quả", result.rows);
  add("Sản phẩm", result.products);
  add("Người bán", result.sellers);
  add("Danh mục ưu tiên", result.data?.categories);
  add("Khu vực ưu tiên", result.data?.states);

  if (result.payment && Object.keys(result.payment).length) add("Thanh toán", [result.payment]);
  if (result.shipping && Object.keys(result.shipping).length) add("Vận chuyển", [result.shipping]);
  if (result.review && Object.keys(result.review).length) add("Đánh giá", [result.review]);
  if (result.customer && Object.keys(result.customer).length) add("Khách hàng", [result.customer]);
  if (result.summary && Object.keys(result.summary).length) add("Tóm tắt", [result.summary]);

  return tables;
}

function normalizeMessage(message) {
  const payload = message.payload || null;
  const text = payload && message.role === "assistant" && looksLikeBadAnalysis(message.text)
    ? answerText(payload)
    : message.text || "";
  return {
    id: message.id || createId(),
    role: message.role,
    text,
    payload,
    tables: message.tables || resultTables(payload),
    error: message.error || (message.role === "assistant" && payload && !payload.ok && Boolean(payload.error)),
    loading: Boolean(message.loading)
  };
}

function normalizeConversation(conversation) {
  return {
    id: conversation.id,
    title: conversation.title || "New chat",
    createdAt: conversation.createdAt || conversation.created_at || Date.now(),
    updatedAt: conversation.updatedAt || conversation.updated_at || Date.now(),
    messages: Array.isArray(conversation.messages)
      ? conversation.messages.map(normalizeMessage)
      : []
  };
}

function TextBlock({ text, error }) {
  return (
    <>
      {String(text || "")
        .split(/\n{2,}/)
        .map((block, blockIndex) => (
          <p key={blockIndex} className={error ? "text-red-200" : ""}>
            {block.split("\n").map((line, lineIndex) => (
              <span key={lineIndex}>
                {lineIndex > 0 && <br />}
                {line}
              </span>
            ))}
          </p>
        ))}
    </>
  );
}

function DataTable({ title, rows }) {
  const visibleRows = rows.slice(0, 12);
  const columns = Array.from(
    visibleRows.reduce((set, row) => {
      Object.keys(row || {}).forEach((key) => set.add(key));
      return set;
    }, new Set())
  );

  if (!columns.length) return null;

  function onWheel(event) {
    if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
    const scroller = event.currentTarget.closest("[data-chat-scroll]");
    if (!scroller || scroller.scrollHeight <= scroller.clientHeight) return;
    event.preventDefault();
    event.stopPropagation();
    scroller.scrollTop += event.deltaY;
  }

  return (
    <div className="mt-3 max-w-full overflow-auto rounded-ui border border-line bg-[#0d0d0d]" onWheel={onWheel}>
      <div className="border-b border-line px-3 py-2 text-sm font-semibold text-zinc-200">
        {title}
      </div>
      <table className="w-full min-w-[620px] border-collapse text-sm">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column} className="sticky top-0 border-b border-line bg-[#181818] px-3 py-2 text-left font-semibold text-zinc-200">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {visibleRows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {columns.map((column) => (
                <td key={column} className="border-b border-[#252525] px-3 py-2 align-top last:border-b-0">
                  {formatNumber(row[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Message({ message }) {
  const isUser = message.role === "user";

  return (
    <article className={`grid w-full max-w-[920px] gap-3 ${isUser ? "grid-cols-[minmax(0,760px)_34px]" : "grid-cols-[34px_minmax(0,820px)]"} mx-auto`}>
      <div className={`${isUser ? "col-start-2 bg-[#2c2c2c]" : "bg-[#19324a]"} grid h-[30px] w-[30px] place-items-center rounded-full text-xs font-bold text-blue-50`}>
        {isUser ? "U" : "OA"}
      </div>
      <div className={`${isUser ? "col-start-1 row-start-1 justify-self-end rounded-[18px] bg-panel px-4 py-2" : "pt-1"} min-w-0 max-w-[650px] overflow-wrap-anywhere text-[15px] leading-relaxed text-zinc-100`}>
        {message.loading ? (
          <span className="inline-flex items-center gap-2 text-muted">
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-700 border-t-white" />
            Đang phân tích
          </span>
        ) : (
          <>
            <TextBlock text={message.text} error={message.error} />
            {(message.tables || []).map((table, index) => (
              <DataTable key={`${table.title}-${index}`} title={table.title} rows={table.rows} />
            ))}
          </>
        )}
      </div>
    </article>
  );
}

function Sidebar({
  conversations,
  activeConversationId,
  onFill,
  onOpenConversation,
  onDeleteConversation,
  onNewChat,
  statusLabel,
  onRefresh,
  open,
  onClose
}) {
  const [historySearch, setHistorySearch] = useState("");
  const searchText = historySearch.trim().toLowerCase();
  const filteredConversations = useMemo(() => {
    if (!searchText) return conversations;
    return conversations.filter((conversation) => {
      const title = conversation.title || "";
      const messageText = (conversation.messages || [])
        .map((message) => message.text || "")
        .join(" ");
      return `${title} ${messageText}`.toLowerCase().includes(searchText);
    });
  }, [conversations, searchText]);

  return (
    <aside className={`${open ? "translate-x-0" : "-translate-x-full"} fixed inset-y-0 left-0 z-20 grid w-[272px] grid-rows-[auto_minmax(0,1fr)_auto] border-r border-[#252525] bg-sidebar transition-transform md:static md:translate-x-0`}>
      <div className="p-3">
        <div className="mb-2 flex min-h-10 items-center justify-between px-2">
          <div className="text-lg font-bold">Olist Agent</div>
          <button className="grid h-10 w-10 place-items-center rounded-full hover:bg-panelHover" type="button" title="Ẩn thanh bên" onClick={onClose}>
            <PanelLeftClose size={18} />
          </button>
        </div>

        <nav className="grid gap-1">
          <button className="flex min-h-10 items-center gap-3 rounded-ui bg-panelHover px-3 text-left" type="button" onClick={onNewChat}>
            <Plus size={18} />
            <span className="truncate text-sm">New chat</span>
          </button>
          <label className="flex min-h-10 items-center gap-3 rounded-ui px-3 text-left text-sm text-zinc-100 hover:bg-panel focus-within:bg-panel">
            <Search size={18} className="shrink-0" />
            <input
              value={historySearch}
              placeholder="Tìm lịch sử"
              className="min-w-0 flex-1 bg-transparent outline-none placeholder:text-zinc-400"
              onChange={(event) => setHistorySearch(event.target.value)}
            />
          </label>
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button key={item.label} className="flex min-h-10 items-center gap-3 rounded-ui px-3 text-left hover:bg-panel" type="button" onClick={() => onFill(item.question)}>
                <Icon size={18} />
                <span className="truncate text-sm">{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      <div className="min-h-0 overflow-y-auto overflow-x-hidden px-3 pb-4">
        <div className="mb-2 mt-3 px-2 text-sm font-semibold">Recents</div>
        <div className="grid gap-1">
          {conversations.length === 0 ? (
            <div className="px-3 py-2 text-sm leading-relaxed text-zinc-500">
              Chưa có lịch sử trò chuyện.
            </div>
          ) : filteredConversations.length === 0 ? (
            <div className="px-3 py-2 text-sm leading-relaxed text-zinc-500">
              Không tìm thấy cuộc trò chuyện.
            </div>
          ) : (
            filteredConversations.map((conversation) => (
              <div
                key={conversation.id}
                className={`grid min-h-9 grid-cols-[minmax(0,1fr)_32px] items-center rounded-ui text-sm hover:bg-panel ${conversation.id === activeConversationId ? "bg-panel text-white" : "text-zinc-200"}`}
              >
                <button
                  className="min-h-9 min-w-0 px-3 text-left"
                  type="button"
                  onClick={() => onOpenConversation(conversation.id)}
                >
                  <span className="block truncate">{conversation.title}</span>
                </button>
                <button
                  className="grid h-8 w-8 place-items-center rounded-md text-zinc-500 hover:bg-panelHover hover:text-red-200"
                  type="button"
                  title="Xóa lịch sử trò chuyện"
                  onClick={() => onDeleteConversation(conversation.id)}
                >
                  <Trash2 size={15} />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="p-3">
        <div className="grid min-h-[50px] grid-cols-[32px_minmax(0,1fr)_40px] items-center gap-3 rounded-ui p-2 hover:bg-panel">
          <div className="grid h-[30px] w-[30px] place-items-center rounded-full bg-[#19324a] text-xs font-bold text-blue-50">OA</div>
          <div className="min-w-0">
            <div className="truncate text-sm font-bold">Olist Analyst</div>
            <div className="text-xs text-muted">{statusLabel}</div>
          </div>
          <button className="grid h-10 w-10 place-items-center rounded-full hover:bg-panelHover" type="button" title="Làm mới trạng thái" onClick={onRefresh}>
            <RefreshCw size={18} />
          </button>
        </div>
      </div>
    </aside>
  );
}

export default function App() {
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [status, setStatus] = useState({ label: "Checking", state: "checking" });
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const abortControllerRef = useRef(null);
  const pendingRequestRef = useRef(null);
  const workspaceRef = useRef(null);
  const textareaRef = useRef(null);

  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeConversationId) || null,
    [activeConversationId, conversations]
  );
  const messages = activeConversation?.messages || [];

  const statusLabel = useMemo(() => {
    if (status.state === "ok") return "MCP active";
    if (status.state === "error") return "MCP error";
    if (status.state === "offline") return "Offline";
    return "Checking";
  }, [status.state]);

  useEffect(() => {
    loadStatus();
    loadHistory();
  }, []);

  useEffect(() => {
    if (workspaceRef.current) workspaceRef.current.scrollTop = workspaceRef.current.scrollHeight;
  }, [messages]);

  useEffect(() => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = "auto";
    textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
  }, [input]);

  async function loadStatus() {
    setStatus({ label: "Checking", state: "checking" });
    try {
      const response = await fetch("/mcp-status");
      const data = await response.json();
      const ok = Boolean(data.ok);
      setStatus({ label: ok ? "Ready" : "Error", state: ok ? "ok" : "error" });
    } catch {
      setStatus({ label: "Offline", state: "offline" });
    }
  }

  async function loadHistory() {
    try {
      const response = await fetch("/conversations");
      const data = await response.json();
      if (!data.ok || !Array.isArray(data.conversations)) return;
      setConversations(data.conversations.map(normalizeConversation));
    } catch {
      setConversations([]);
    }
  }

  function upsertConversation(conversation) {
    const normalized = normalizeConversation(conversation);
    setConversations((current) => {
      const next = [
        normalized,
        ...current.filter((item) => item.id !== normalized.id)
      ].sort((a, b) => timeValue(b.updatedAt) - timeValue(a.updatedAt));
      return next.slice(0, 30);
    });
  }

  function ensureConversation(question) {
    if (activeConversation) return activeConversation;
    const conversation = createConversation(titleFromQuestion(question));
    setActiveConversationId(conversation.id);
    upsertConversation(conversation);
    return conversation;
  }

  function fillQuestion(question) {
    setInput(question);
    setSidebarOpen(false);
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  function newChat() {
    setActiveConversationId(null);
    setInput("");
    setSidebarOpen(false);
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  function openConversation(conversationId) {
    setActiveConversationId(conversationId);
    setSidebarOpen(false);
    requestAnimationFrame(() => workspaceRef.current?.focus());
  }

  function replaceLoadingMessage(conversationId, loadingId, replacement) {
    setConversations((current) =>
      current.map((item) =>
        item.id === conversationId
          ? {
              ...item,
              messages: item.messages.map((message) =>
                message.id === loadingId ? replacement : message
              ),
              updatedAt: Date.now()
            }
          : item
      )
    );
  }

  function stopAnalysis() {
    const controller = abortControllerRef.current;
    const pendingRequest = pendingRequestRef.current;
    if (!controller || !pendingRequest) return;

    controller.abort();
    abortControllerRef.current = null;
    pendingRequestRef.current = null;
    setPending(false);
    replaceLoadingMessage(pendingRequest.conversationId, pendingRequest.loadingId, {
      id: pendingRequest.loadingId,
      role: "assistant",
      text: "Đã tạm dừng phân tích."
    });
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  async function deleteConversation(conversationId) {
    const confirmed = window.confirm("Xóa cuộc trò chuyện này khỏi lịch sử?");
    if (!confirmed) return;

    if (pendingRequestRef.current?.conversationId === conversationId) {
      stopAnalysis();
    }

    setConversations((current) => current.filter((conversation) => conversation.id !== conversationId));
    if (conversationId === activeConversationId) {
      setActiveConversationId(null);
    }

    try {
      await fetch(`/conversations/${encodeURIComponent(conversationId)}`, {
        method: "DELETE"
      });
    } catch {
      loadHistory();
    }
  }

  async function submit(question) {
    const clean = question.trim();
    if (!clean || pending) return;

    setInput("");
    setPending(true);

    const loadingId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
    const conversation = ensureConversation(clean);
    const title = conversation.title === "New chat" ? titleFromQuestion(clean) : conversation.title;
    const nextMessages = [
      ...conversation.messages,
      { id: `${loadingId}-user`, role: "user", text: clean },
      { id: loadingId, role: "assistant", loading: true, text: "" }
    ];
    upsertConversation({
      ...conversation,
      title,
      messages: nextMessages,
      updatedAt: Date.now()
    });

    const controller = new AbortController();
    abortControllerRef.current = controller;
    pendingRequestRef.current = { conversationId: conversation.id, loadingId };

    try {
      const response = await fetch("/ask-via-mcp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          question: clean,
          output_path: "",
          conversation_id: conversation.id
        })
      });
      const result = await response.json();
      if (controller.signal.aborted) return;
      if (result.conversation) {
        const savedConversation = normalizeConversation(result.conversation);
        setActiveConversationId(savedConversation.id);
        upsertConversation(savedConversation);
      } else {
        setConversations((current) =>
          current.map((item) =>
            item.id === conversation.id
              ? {
                  ...item,
                  messages: item.messages.map((message) =>
                    message.id === loadingId
                      ? {
                          id: loadingId,
                          role: "assistant",
                          text: answerText(result),
                          tables: resultTables(result),
                          error: !result.ok && Boolean(result.error)
                        }
                      : message
                  ),
                  updatedAt: Date.now()
                }
              : item
          )
        );
      }
    } catch (error) {
      if (error.name === "AbortError") return;
      setConversations((current) =>
        current.map((item) =>
          item.id === conversation.id
            ? {
                ...item,
                messages: item.messages.map((message) =>
                  message.id === loadingId
                    ? {
                        id: loadingId,
                        role: "assistant",
                        text: `Không gọi được MCP server: ${error.message}`,
                        error: true
                      }
                    : message
                ),
                updatedAt: Date.now()
              }
            : item
        )
      );
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
        pendingRequestRef.current = null;
        setPending(false);
      }
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }

  function onSubmit(event) {
    event.preventDefault();
    submit(input);
  }

  function onKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit(input);
    }
  }

  return (
    <div className="grid h-screen min-h-0 overflow-hidden bg-page text-zinc-50 md:grid-cols-[272px_minmax(0,1fr)]">
      <Sidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        onFill={fillQuestion}
        onOpenConversation={openConversation}
        onDeleteConversation={deleteConversation}
        onNewChat={newChat}
        statusLabel={statusLabel}
        onRefresh={loadStatus}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="grid min-h-0 min-w-0 grid-rows-[auto_minmax(0,1fr)_auto] overflow-hidden">
        <header className="flex min-h-[50px] shrink-0 items-center justify-between px-4 py-2">
          <button className="grid h-10 w-10 place-items-center rounded-full hover:bg-panelHover md:hidden" type="button" title="Mở thanh bên" onClick={() => setSidebarOpen(true)}>
            <Menu size={18} />
          </button>
          <div className="hidden h-10 w-10 md:block" />
          <div className="inline-flex min-w-28 items-center justify-end gap-2 text-sm text-muted">
            <span>{status.label}</span>
            <span className={`h-2.5 w-2.5 rounded-full ${status.state === "ok" ? "bg-accent" : status.state === "checking" ? "bg-muted" : "bg-red-500"}`} />
          </div>
        </header>

        <section ref={workspaceRef} className="relative min-h-0 overflow-y-scroll overflow-x-hidden overscroll-contain outline-none" data-chat-scroll tabIndex={-1}>
          {messages.length === 0 && (
            <div className="pointer-events-none absolute inset-0 grid place-items-center p-7">
              <div className="-translate-y-5 grid w-full max-w-[780px] justify-items-center gap-6">
                <h1 className="m-0 text-center text-2xl font-semibold">Ready when you are.</h1>
                <div className="pointer-events-auto flex flex-wrap justify-center gap-3">
                  {quickActions.map((item) => {
                    const Icon = item.icon;
                    return (
                      <button key={item.label} className="inline-flex min-h-10 items-center gap-2 rounded-full border border-line px-4 text-sm hover:bg-panel" type="button" onClick={() => fillQuestion(item.question)}>
                        <Icon size={18} />
                        {item.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          <div className="flex min-h-full flex-col gap-5 px-5 py-3 pb-8">
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
          </div>
        </section>

        <section className="grid shrink-0 justify-items-center gap-3 px-5 pb-6 pt-3">
          <form className="grid min-h-[58px] w-full max-w-[780px] grid-cols-[auto_minmax(0,1fr)_auto_auto] items-end gap-2 rounded-[30px] bg-panel p-2 max-md:grid-cols-[auto_minmax(0,1fr)_auto]" onSubmit={onSubmit}>
            <button className="grid h-10 w-10 place-items-center rounded-full hover:bg-panelHover" type="button" title="Câu hỏi mẫu" onClick={() => fillQuestion("năm 2018, nếu muốn cải thiện trải nghiệm khách hàng, nên ưu tiên danh mục hoặc khu vực nào?")}>
              <Plus size={18} />
            </button>
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              placeholder="Ask anything"
              disabled={pending}
              className="max-h-40 min-h-[42px] w-full resize-none bg-transparent py-2.5 text-[15px] leading-snug outline-none placeholder:text-muted"
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={onKeyDown}
            />
            <div className="flex h-9 select-none items-center rounded-full px-3 text-sm text-zinc-300 max-md:hidden" title="Mô hình trả lời: Gemma local">
              Gemma local
            </div>
            {pending ? (
              <button className="grid h-10 w-10 place-items-center rounded-full bg-white text-black" type="button" title="Tạm dừng phân tích" onClick={stopAnalysis}>
                <Square size={15} fill="currentColor" />
              </button>
            ) : (
              <button className="grid h-10 w-10 place-items-center rounded-full bg-white text-black disabled:cursor-not-allowed disabled:opacity-50" type="submit" disabled={!input.trim()} title="Gửi">
                <ArrowUp size={18} />
              </button>
            )}
          </form>
          <div className="w-full max-w-[780px] text-center text-xs text-zinc-500">
            Olist Agent có thể sai. Hãy kiểm tra số liệu quan trọng trước khi sử dụng.
          </div>
        </section>
      </main>
    </div>
  );
}
