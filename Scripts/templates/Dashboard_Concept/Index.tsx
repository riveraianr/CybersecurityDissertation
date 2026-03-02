import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import VulnerabilityDashboard from "@/components/VulnerabilityDashboard";
import ChatInterface from "@/components/ChatInterface";
import ThemeToggle from "@/components/ThemeToggle";
import { Search, Filter, BarChart3, ShieldAlert, Info } from "lucide-react";
import Sidebar from "@/components/Sidebar";

const Index = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  
  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <Sidebar isOpen={isMenuOpen} setIsOpen={setIsMenuOpen} />
      
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="border-b bg-card">
          <div className="flex items-center justify-between p-4">
            <div className="flex items-center gap-2">
              <Button 
                variant="ghost" 
                size="icon" 
                onClick={() => setIsMenuOpen(!isMenuOpen)}
              >
                <Filter className="h-5 w-5" />
                <span className="sr-only">Toggle Menu</span>
              </Button>
              <h1 className="text-xl font-bold text-primary">
                RA5olver <span className="hidden sm:inline">Vulnerability Dashboard</span>
              </h1>
            </div>
            
            <div className="flex items-center gap-2">
              <div className="relative max-w-sm">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input 
                  type="search" 
                  placeholder="Search vulnerabilities..." 
                  className="pl-8 w-[200px] lg:w-[300px]" 
                />
              </div>
              <ThemeToggle />
            </div>
          </div>
        </header>
        
        <main className="flex-1 overflow-y-auto p-4">
          <div className="container mx-auto space-y-4 py-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Critical Vulnerabilities</CardDescription>
                  <CardTitle className="text-2xl">24</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xs text-muted-foreground flex items-center">
                    <ShieldAlert className="mr-1 h-4 w-4 text-destructive" />
                    <span className="text-destructive font-medium">+7</span>&nbsp;since last scan
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>High Priority</CardDescription>
                  <CardTitle className="text-2xl">36</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xs text-muted-foreground flex items-center">
                    <ShieldAlert className="mr-1 h-4 w-4 text-orange-500" />
                    <span>12 require immediate attention</span>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Active Exploits</CardDescription>
                  <CardTitle className="text-2xl">8</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xs text-muted-foreground flex items-center">
                    <Info className="mr-1 h-4 w-4 text-blue-500" />
                    <span>3 with known mitigations</span>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Remediation Progress</CardDescription>
                  <CardTitle className="text-2xl">63%</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xs text-muted-foreground flex items-center">
                    <BarChart3 className="mr-1 h-4 w-4 text-green-500" />
                    <span className="text-green-500 font-medium">+12%</span>&nbsp;from last week
                  </div>
                </CardContent>
              </Card>
            </div>
            
            <Tabs defaultValue="vulnerabilities" className="w-full">
              <TabsList className="grid w-full grid-cols-3 md:w-auto md:inline-flex">
                <TabsTrigger value="vulnerabilities">Vulnerabilities</TabsTrigger>
                <TabsTrigger value="analytics">Analytics</TabsTrigger>
                <TabsTrigger value="reports">Reports</TabsTrigger>
              </TabsList>
              
              <TabsContent value="vulnerabilities" className="mt-4">
                <VulnerabilityDashboard />
              </TabsContent>
              
              <TabsContent value="analytics" className="mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Vulnerability Analytics</CardTitle>
                    <CardDescription>
                      Advanced analytics and insights for your vulnerability data
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">
                      Analytics content will be displayed here with charts showing trends, 
                      distribution of vulnerabilities by severity, and other key metrics.
                    </p>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="reports" className="mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Vulnerability Reports</CardTitle>
                    <CardDescription>
                      Generate and view vulnerability reports
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">
                      Reports content will allow generating customized reports for different stakeholders, 
                      including executive summaries, detailed technical reports, and compliance documentation.
                    </p>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </main>
      </div>
      
      <ChatInterface />
    </div>
  );
};

export default Index;
