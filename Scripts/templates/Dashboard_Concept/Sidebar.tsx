import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { Search, BarChart3, AlertCircle, Shield, ShieldAlert, Settings, FileText, Clock, Filter } from "lucide-react";

interface SidebarProps {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
}

const Sidebar = ({ isOpen, setIsOpen }: SidebarProps) => {
  const [activeSection, setActiveSection] = useState("dashboard");
  
  const sidebarItems = [
    {
      id: "dashboard",
      label: "Dashboard",
      icon: <BarChart3 className="h-5 w-5" />
    },
    {
      id: "vulnerabilities",
      label: "Vulnerabilities",
      icon: <ShieldAlert className="h-5 w-5" />
    },
    {
      id: "search",
      label: "Advanced Search",
      icon: <Search className="h-5 w-5" />
    },
    {
      id: "alerts",
      label: "Alerts",
      icon: <AlertCircle className="h-5 w-5" />
    },
    {
      id: "reports",
      label: "Reports",
      icon: <FileText className="h-5 w-5" />
    },
    {
      id: "history",
      label: "Scan History",
      icon: <Clock className="h-5 w-5" />
    }
  ];
  
  const statistics = [
    { label: "Critical", value: 24, color: "bg-red-500" },
    { label: "High", value: 36, color: "bg-orange-500" },
    { label: "Medium", value: 45, color: "bg-yellow-500" },
    { label: "Low", value: 18, color: "bg-green-500" }
  ];
  
  return (
    <aside
      className={cn(
        "fixed inset-y-0 left-0 z-20 flex w-72 flex-col bg-sidebar border-r border-sidebar-border transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0",
        isOpen ? "translate-x-0" : "-translate-x-full"
      )}
    >
      <div className="bg-sidebar-accent p-4 border-b border-sidebar-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Shield className="h-6 w-6 text-primary" />
          <span className="font-bold text-lg text-sidebar-foreground">RA5olver</span>
        </div>
        <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)} className="lg:hidden">
          <Filter className="h-5 w-5" />
        </Button>
      </div>
      
      <div className="bg-sidebar-accent p-4 border-b border-sidebar-border">
        <div className="space-y-1">
          <h3 className="text-sm font-medium text-sidebar-foreground">Vulnerability Summary</h3>
          <div className="grid grid-cols-2 gap-2">
            {statistics.map((stat) => (
              <div key={stat.label} className="flex items-center p-2 rounded-md bg-sidebar border border-sidebar-border">
                <div className={`h-2 w-2 rounded-full mr-2 ${stat.color}`}></div>
                <span className="text-xs text-sidebar-foreground">{stat.label}</span>
                <span className="ml-auto font-medium text-sidebar-foreground">{stat.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      <ScrollArea className="flex-1">
        <div className="px-3 py-2">
          <div className="space-y-1">
            <p className="text-xs font-medium text-sidebar-foreground/70 px-4 py-2">
              Main Navigation
            </p>
            <nav className="grid gap-1">
              {sidebarItems.map((item) => (
                <Button
                  key={item.id}
                  variant="ghost"
                  className={cn(
                    "group flex h-10 w-full items-center justify-start gap-3 rounded-md px-3 py-2 text-sm font-medium hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                    activeSection === item.id && "bg-sidebar-accent text-sidebar-accent-foreground"
                  )}
                  onClick={() => setActiveSection(item.id)}
                >
                  {item.icon}
                  <span>{item.label}</span>
                </Button>
              ))}
            </nav>
          </div>
        </div>
      </ScrollArea>
      
      <div className="bg-sidebar-accent border-t border-sidebar-border p-4">
        <Button 
          variant="outline" 
          className="w-full flex items-center gap-2 bg-sidebar border-sidebar-border"
        >
          <Settings className="h-4 w-4" />
          <span>Settings</span>
        </Button>
      </div>
    </aside>
  );
};

export default Sidebar;
