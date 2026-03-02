import { useState, useRef, useEffect } from "react";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { MessageSquare, Send, X, Maximize2, Minimize2 } from "lucide-react";
import { MOCK_VULNERABILITIES } from "@/data/mockData";

interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  timestamp: Date;
}

const INITIAL_MESSAGES: Message[] = [
  {
    id: "1",
    type: "bot",
    content: "Hello! I'm RA5olver's assistant. How can I help you analyze vulnerabilities today?",
    timestamp: new Date()
  }
];

const ChatInterface = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [input, setInput] = useState("");
  const [selectedCve, setSelectedCve] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSendMessage = () => {
    if (!input.trim()) return;
    
    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    
    // Simulate bot response with context-awareness
    setTimeout(() => {
      let responseContent = "";
      
      if (selectedCve) {
        const vuln = MOCK_VULNERABILITIES.find(v => v.cve_id === selectedCve);
        if (vuln) {
          if (input.toLowerCase().includes("severity") || input.toLowerCase().includes("how serious")) {
            responseContent = `${selectedCve} has a ${vuln.reported_severity} severity rating. ${vuln.severity_rationale}`;
          } else if (input.toLowerCase().includes("fix") || input.toLowerCase().includes("remediat") || input.toLowerCase().includes("mitigat")) {
            responseContent = `To address ${selectedCve}, you should: ${vuln.remediation_steps}`;
          } else if (input.toLowerCase().includes("priority") || input.toLowerCase().includes("important")) {
            responseContent = `${selectedCve} ${vuln.priority === "Yes" ? "is" : "is not"} a priority. ${vuln.priority_justification}`;
          } else if (input.toLowerCase().includes("exploit") || input.toLowerCase().includes("attack")) {
            responseContent = vuln.has_exploit 
              ? `Yes, ${selectedCve} has known exploits in the wild. This increases its risk significantly.` 
              : `No known exploits for ${selectedCve} have been reported yet, but you should still address it according to its severity.`;
          } else {
            responseContent = `Regarding ${selectedCve}: ${vuln.analysis}`;
          }
        } else {
          responseContent = `I don't have specific information about ${selectedCve}. Please select a different CVE ID or ask a general question.`;
        }
      } else {
        if (input.toLowerCase().includes("most critical") || input.toLowerCase().includes("highest priority")) {
          const criticalVulns = MOCK_VULNERABILITIES.filter(v => v.reported_severity.toLowerCase() === "critical" || v.priority === "Yes");
          responseContent = `The most critical vulnerabilities are: ${criticalVulns.slice(0, 3).map(v => v.cve_id).join(", ")}. I recommend addressing these first.`;
        } else if (input.toLowerCase().includes("summary") || input.toLowerCase().includes("overview")) {
          responseContent = `Current summary: There are ${MOCK_VULNERABILITIES.length} vulnerabilities in the system. ${MOCK_VULNERABILITIES.filter(v => v.reported_severity.toLowerCase() === "critical").length} critical, ${MOCK_VULNERABILITIES.filter(v => v.reported_severity.toLowerCase() === "high").length} high, and ${MOCK_VULNERABILITIES.filter(v => v.has_exploit).length} with known exploits.`;
        } else {
          responseContent = "I can help you analyze vulnerabilities, understand their impact, and prioritize remediation efforts. For specific details, please select a CVE ID from the dropdown.";
        }
      }
      
      const botMessage: Message = {
        id: Date.now().toString(),
        type: "bot",
        content: responseContent,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, botMessage]);
    }, 1000);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  // Chat window that can be toggled and expanded
  const renderChatWindow = () => {
    return (
      <div className={`fixed bottom-4 right-4 z-50 flex flex-col ${
        isExpanded ? "w-[600px] h-[80vh]" : "w-[350px] h-[500px]"
      } bg-card border rounded-lg shadow-lg transition-all duration-200 ease-in-out`}>
        <div className="flex items-center justify-between p-3 border-b">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5 text-primary" />
            <h3 className="font-medium">RA5olver Assistant</h3>
          </div>
          <div className="flex items-center">
            <Button variant="ghost" size="icon" onClick={() => setIsExpanded(!isExpanded)}>
              {isExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </Button>
            <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div 
              key={message.id} 
              className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}
            >
              <div className={`max-w-[80%] rounded-lg px-4 py-2 ${
                message.type === "user" 
                  ? "bg-primary text-primary-foreground" 
                  : "bg-muted/50 text-foreground"
              }`}>
                <p className="text-sm">{message.content}</p>
                <p className="text-xs text-muted-foreground/70 mt-1">
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="p-3 border-t">
          <div className="flex gap-2 mb-2">
            <Select value={selectedCve} onValueChange={setSelectedCve}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select CVE (optional)" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All Vulnerabilities</SelectItem>
                {MOCK_VULNERABILITIES.map(vuln => (
                  <SelectItem key={vuln.cve_id} value={vuln.cve_id}>
                    {vuln.cve_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          <div className="flex gap-2">
            <Textarea 
              placeholder="Type your question..."
              className="min-h-[60px]"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <Button onClick={handleSendMessage} className="self-end">
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    );
  };

  // Chat button that opens the chat window
  return (
    <>
      {isOpen ? (
        renderChatWindow()
      ) : (
        <Button 
          className="fixed bottom-4 right-4 rounded-full h-12 w-12 shadow-lg z-50"
          onClick={() => setIsOpen(true)}
        >
          <MessageSquare className="h-5 w-5" />
        </Button>
      )}
      
      <Sheet open={isOpen && !isExpanded} onOpenChange={(open) => setIsOpen(open)}>
        <SheetContent side="right" className="w-[400px] sm:w-[540px] p-0">
          {renderChatWindow()}
        </SheetContent>
      </Sheet>
    </>
  );
};

export default ChatInterface;
